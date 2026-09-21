# TCNN
python main.py --run-name tcnn3 \
  --override model.name=texture_cnn \
  --override model.depth=3 \
  --override model.pretrained=false \
  --override model.dropout=0.5 \
  --override model.fc_dim=4096 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override augmentation.gaussian_noise.std=0.01

# TwistNet18
python main.py --run-name twistnet18 \
  --override model.name=twistnet18 \
  --override model.pretrained=false \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override augmentation.gaussian_noise.std=0.01

# HiPerViT-S: DINOv2 ViT-S/14, блоки 1 и 5, interact → distill.
# Голова (SRM + Perceiver) учится с train.lr, ViT — с train.backbone_lr.
python main.py --run-name hipervit_s \
  --override model.name=hipervit \
  --override model.pretrained=true \
  --override model.backbone=vit_small_patch14_dinov2.lvd142m \
  --override model.stages=[1,5] \
  --override model.sketch_dim=512 \
  --override model.num_latents=64 \
  --override model.topology=interact_distill \
  --override model.dropout=0.0 \
  --override model.drop_path_rate=0.0 \
  --override train.lr=3e-4 \
  --override train.backbone_lr=3e-5 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override augmentation.gaussian_noise.std=0.01

# VORTEX-B: замороженный ViT-B/16 (IN-21k), токены всех блоков, суп из 16 RAE.
# Учится только линейная голова (в статье на её месте линейный SVM).
python main.py --run-name vortex_b \
  --override model.name=vortex \
  --override model.pretrained=true \
  --override model.backbone=vit_base_patch16_224.orig_in21k \
  --override model.m=16 \
  --override model.rae_hidden=1 \
  --override model.dropout=0.0 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override augmentation.gaussian_noise.std=0.01