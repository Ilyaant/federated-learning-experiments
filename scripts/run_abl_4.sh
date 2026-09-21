# P3-A: rot90-победитель + sa12 (sa12 ещё не тестировали с degrees=0)
python main.py --run-name abl_p3_rot90_sa12 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override model.name=fastvit_sa12

# P3-C: слабее шум (0 выключил clear, 0.03 может быть грубым)
python main.py --run-name abl_p3_noise01 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override augmentation.gaussian_noise.std=0.01

# P3-D: 256 + rot90 (прошлый size256 убит и шёл с degrees=15)
python main.py --run-name abl_p3_size256 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override patches.size=256 \
  --override train.num_workers=0 \
  --override evaluation.num_workers=0

python main.py --run-name abl_p3_size256_noise01 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0 \
  --override patches.size=256 \
  --override train.num_workers=0 \
  --override evaluation.num_workers=0 \
  --override augmentation.gaussian_noise.std=0.01