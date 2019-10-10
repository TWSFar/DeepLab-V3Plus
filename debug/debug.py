from tensorboardX import SummaryWriter
writer = SummaryWriter('debug')
writer.add_scalar('Train/Loss', 3, 20)
writer.add_scalar('Test/Accu', 10, 30)

