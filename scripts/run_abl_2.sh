## Второй запуск абляционного тестирования (по итогам первого)

# Оверлап, без blur
python main.py --run-name abl_overlap_train05_no_blur \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0

python main.py --run-name abl_p1_lr \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override train.lr=3.0e-5

python main.py --run-name abl_p1_backbone_lr \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override train.backbone_lr=1.0e-5

python main.py --run-name abl_p1_reg \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override model.dropout=0.35 \
  --override model.drop_path_rate=0.2

python main.py --run-name abl_p1_ls0 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override train.label_smoothing=0