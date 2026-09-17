## Первый запуск абляционного тестирования

# denser train grid, сравнимый test
python main.py --run-name abl_overlap_train05 \
  --override patches.train_overlap=0.5

# denser train grid, test
python main.py --run-name abl_overlap_train05_eval05 \
  --override patches.train_overlap=0.5 \
  --override patches.eval_overlap=0.5

# гипотеза: blur убивает текстуру
python main.py --run-name abl_no_blur \
  --override augmentation.gaussian_blur.prob=0

# clear недопредставлен (59 vs 106 GP)
python main.py --run-name abl_class_w \
  --override train.class_weights=true

# без аугментаций
python main.py --run-name no_augmentations \
  --override augmentation.enabled=false