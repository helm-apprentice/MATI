import unittest
from ce_utils import adjust_keep_rate

class TestAdjustKeepRate(unittest.TestCase):

    def test_during_warmup(self):
        """测试在warmup阶段时的keep rate"""
        epoch = 2
        warmup_epochs = 5
        total_epochs = 10
        ITERS_PER_EPOCH = 100
        expected_keep_rate = 1.0  # 在warmup阶段，keep rate应为1
        self.assertEqual(adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH), expected_keep_rate)

    def test_after_total_epochs(self):
        """测试在总epoch数之后时的keep rate"""
        epoch = 12
        warmup_epochs = 5
        total_epochs = 10
        ITERS_PER_EPOCH = 100
        base_keep_rate = 0.5
        expected_keep_rate = base_keep_rate  # 当epoch大于total_epochs时，keep rate应为base_keep_rate
        self.assertEqual(adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate), expected_keep_rate)

    def test_custom_parameters(self):
        """测试使用自定义参数时的keep rate"""
        epoch = 7
        warmup_epochs = 5
        total_epochs = 10
        ITERS_PER_EPOCH = 100
        base_keep_rate = 0.3
        max_keep_rate = 0.8
        expected_keep_rate = 0.5331711237247449  # 预期计算结果
        self.assertAlmostEqual(adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate, max_keep_rate), expected_keep_rate, places=6)

    def test_specific_iter(self):
        """测试在指定iter时的keep rate"""
        epoch = 3
        warmup_epochs = 5
        total_epochs = 10
        ITERS_PER_EPOCH = 100
        iters = 250  # 指定的iter数
        base_keep_rate = 0.5
        max_keep_rate = 1
        expected_keep_rate = 0.7015625  # 预期计算结果
        self.assertAlmostEqual(adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate, max_keep_rate, iters), expected_keep_rate, places=6)

if __name__ == '__main__':
    unittest.main()