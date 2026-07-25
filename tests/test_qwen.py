"""Test suite for Qwen VL Flash Router solution."""
import unittest
from qwen_vl_flash_router import QwenVLFlashRouter

class TestQwenVLFlashRouter(unittest.TestCase):

    def test_visual_packing(self):
        router = QwenVLFlashRouter()
        res = router.pack_visual_and_route(
            image_resolutions=[(1024, 768), (512, 512)], text_token_count=4096
        )
        
        self.assertEqual(res["status"], "FLASH_ROUTER_OPTIMAL")
        self.assertTrue(res["total_tokens"] > 0)

if __name__ == "__main__":
    unittest.main()
