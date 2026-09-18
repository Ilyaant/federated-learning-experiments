# P2-A: только rot90, без свободного поворота (не меняет масштаб текстуры)
python main.py --run-name abl_p2_rot90 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.rotation.degrees=0

# P2-B: без гауссова шума (та же гипотеза, что и для blur)
python main.py --run-name abl_p2_no_noise \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override augmentation.gaussian_noise.prob=0

# P2-D: нативный вход FastViT 256 вместо 224
python main.py --run-name abl_p2_size256 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override patches.size=256

# P2-C: больше ёмкость (M/clear уже не растут от данных)
python main.py --run-name abl_p2_t8_dist \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override model.name=fastvit_t8.apple_dist_in1k

# P2-C: больше ёмкость (M/clear уже не растут от данных)
python main.py --run-name abl_p2_t12 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override model.name=fastvit_t12

# P2-C: больше ёмкость (M/clear уже не растут от данных)
python main.py --run-name abl_p2_s12 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override model.name=fastvit_s12

# P2-C: больше ёмкость (M/clear уже не растут от данных)
python main.py --run-name abl_p2_sa12 \
  --override patches.train_overlap=0.5 \
  --override augmentation.gaussian_blur.prob=0 \
  --override model.name=fastvit_sa12