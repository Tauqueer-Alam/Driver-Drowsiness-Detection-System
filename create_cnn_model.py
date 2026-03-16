import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

def create_cnn():
    """
    Creates a Convolutional Neural Network (CNN) designed specifically to 
    classify image crops of a human eye as Open (1) or Closed (0).
    """
    model = Sequential([

        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("===========================================")
    print("Building Deep Learning Model Architecture...")
    print("===========================================")
    model = create_cnn()
    model.summary()
    

    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"\n[ERROR] Directory '{dataset_path}' not found!")
        print("Please ensure your dataset is placed in a folder named 'dataset'")
        print("with two subfolders: 'Closed_Eyes' and 'Open_Eyes'.")
        exit()
        
    print("\nPreparing Image Data Generators...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 
    )

    batch_sz = 32
    img_sz = (64, 64)
    
    print("Loading Training Data...")
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_sz,
        color_mode="grayscale",
        batch_size=batch_sz,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    
    print("Loading Validation Data...")
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_sz,
        color_mode="grayscale",
        batch_size=batch_sz,
        class_mode="categorical",
        subset="validation",
        shuffle=True
    )
    
 
    print(f"\nClass Indices Mapped: {train_generator.class_indices}")

    checkpoint = ModelCheckpoint(
        "drowsiness_cnn_model.keras", 
        monitor="val_accuracy", 
        verbose=1, 
        save_best_only=True, 
        mode="max"
    )
    

    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=5, 
        verbose=1, 
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
    
    
    EPOCHS = 20 
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )
    