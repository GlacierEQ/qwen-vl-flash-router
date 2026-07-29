/**
 * Qwen VL Flash Router — Multi-Modal Request Router & Attention Budget Manager
 * Routes vision-language requests across heterogeneous GPU pools with
 * token-aware load balancing and flash attention budget enforcement.
 */

export interface VLRequest {
  id: string;
  imageTokens: number;
  textTokens: number;
  modality: 'text' | 'image' | 'video' | 'multimodal';
  priority: 'realtime' | 'batch' | 'background';
  maxLatencyMs: number;
}

export interface GPUNode {
  nodeId: string;
  gpuMemoryGB: number;
  usedMemoryGB: number;
  activeStreams: number;
  flashAttnCapable: boolean;
  vramBandwidthGBps: number;
}

export interface RoutingDecision {
  requestId: string;
  targetNode: string;
  estimatedLatencyMs: number;
  attentionBudgetTokens: number;
  flashAttnEnabled: boolean;
}

export class FlashAttentionBudget {
  private maxTokens: number;
  private currentTokens: number;
  private windowSize: number;

  constructor(maxTokens: number = 131072, windowSize: number = 32768) {
    this.maxTokens = maxTokens;
    this.currentTokens = 0;
    this.windowSize = windowSize;
  }

  allocate(imageTokens: number, textTokens: number): number {
    const totalRequested = imageTokens + textTokens;
    if (totalRequested <= this.windowSize) {
      this.currentTokens += totalRequested;
      return totalRequested;
    }
    // Sliding window: prioritize text context, compress image tokens
    const compressedImage = Math.ceil(imageTokens * 0.25); // 4x image token compression
    const budget = Math.min(compressedImage + textTokens, this.maxTokens);
    this.currentTokens += budget;
    return budget;
  }

  pressure(): number {
    return this.currentTokens / this.maxTokens;
  }

  release(tokens: number): void {
    this.currentTokens = Math.max(0, this.currentTokens - tokens);
  }
}

export class QwenVLFlashRouter {
  private nodes: Map<string, GPUNode> = new Map();
  private budget: FlashAttentionBudget;
  private routingHistory: RoutingDecision[] = [];
  private totalRouted: number = 0;

  constructor(maxBudgetTokens: number = 131072) {
    this.budget = new FlashAttentionBudget(maxBudgetTokens);
  }

  registerNode(node: GPUNode): void {
    this.nodes.set(node.nodeId, node);
  }

  private scoreNode(node: GPUNode, req: VLRequest): number {
    const memPressure = node.usedMemoryGB / node.gpuMemoryGB;
    const streamPenalty = node.activeStreams * 0.05;
    const flashBonus = node.flashAttnCapable && req.imageTokens > 1024 ? -0.2 : 0;
    const bandwidthScore = 1.0 - Math.min(node.vramBandwidthGBps / 3200, 1.0);

    // Lower score = better node
    return memPressure * 0.4 + streamPenalty * 0.2 + flashBonus + bandwidthScore * 0.2;
  }

  route(request: VLRequest): RoutingDecision | null {
    if (this.nodes.size === 0) return null;

    let bestNode: GPUNode | null = null;
    let bestScore = Infinity;

    for (const node of this.nodes.values()) {
      const score = this.scoreNode(node, request);
      if (score < bestScore) {
        bestScore = score;
        bestNode = node;
      }
    }

    if (!bestNode) return null;

    const budgetTokens = this.budget.allocate(request.imageTokens, request.textTokens);
    const estimatedLatency = (budgetTokens / 1000) * (bestNode.flashAttnCapable ? 0.8 : 2.5);

    const decision: RoutingDecision = {
      requestId: request.id,
      targetNode: bestNode.nodeId,
      estimatedLatencyMs: estimatedLatency,
      attentionBudgetTokens: budgetTokens,
      flashAttnEnabled: bestNode.flashAttnCapable,
    };

    bestNode.activeStreams++;
    bestNode.usedMemoryGB += budgetTokens * 2 * 128 / (1024 * 1024 * 1024);
    this.routingHistory.push(decision);
    this.totalRouted++;

    return decision;
  }

  stats(): Record<string, number> {
    return {
      totalRouted: this.totalRouted,
      activeNodes: this.nodes.size,
      budgetPressure: this.budget.pressure(),
    };
  }
}
