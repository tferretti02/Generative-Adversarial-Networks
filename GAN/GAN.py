import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter()


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            # nn.Linear(hidden_dim, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)


def get_noise(n_samples, z_dim, device="cuda"):
    return torch.randn(n_samples, z_dim, device=device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)


# Parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
z_dim = 64
display_step = 500
batch_size = 32
lr = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("You are using", device)

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST(".", download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise).detach()

    fake_preds = disc(fake)
    fake_target = torch.zeros_like(fake_preds, device=device)
    fake_loss = criterion(fake_preds, fake_target)

    real_preds = disc(real)
    real_target = torch.ones_like(real_preds, device=device)
    real_loss = criterion(real_preds, real_target)

    disc_loss = (fake_loss + real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    pred = disc(fake)
    target = torch.ones_like(pred, device=device)
    gen_loss = criterion(pred, target)
    return gen_loss


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False
for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        # Discriminatr
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(
            gen, disc, criterion, real, cur_batch_size, z_dim, device
        )
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        # Average losses
        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        # TensorBoard
        if cur_step % display_step == 0 and cur_step > 0:
            writer.add_scalar("Loss/Generator", mean_generator_loss, cur_step)
            writer.add_scalar("Loss/Discriminator", mean_discriminator_loss, cur_step)
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            writer.add_images("Fake Images", fake.view(-1, 1, 28, 28), cur_step)
            writer.add_images("Real Images", real.view(-1, 1, 28, 28), cur_step)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

writer.close()
