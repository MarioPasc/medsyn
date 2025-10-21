from ultralytics import YOLO

# Model loading

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Model training
results = model.train(data = "C:\Users\User\MyDocs\Universidad\CUARTO_2026\Almacenes_de_datos\proyecto\medsyn\PathMNIST\yolo_cls_dataset", 
                      epochs = 100, imgsz = 64)

