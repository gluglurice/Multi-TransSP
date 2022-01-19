import torch


class LambdaLR:
    """
    学习率衰减
    epochs: epochs数
    offset: 0 初始epoch
    decay_start_epoch: 开始衰减的epoch
    """

    def __init__(self, n_epochs, offset, decay_start_epoch):
        # 开始衰减时的epoch必须在训练结束前
        assert ((n_epochs - decay_start_epoch) > 0), 'Decay must start before the training session ends!'
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
