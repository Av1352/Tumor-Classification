from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(config):
    base_model = ResNet50(
        weights=config['model']['pretrained_weights'],
        include_top=False,
        input_shape=(*config['data']['image_size'], 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(config['model']['dropout_rate'])(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=config['model']['learning_rate']),
                loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_model

def train_model(model, train_dataset, val_dataset, train_generator, val_generator, config):
    steps_per_epoch = train_generator.samples // config['data']['batch_size']
    val_steps = val_generator.samples // config['data']['batch_size']
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config['model']['epochs'],
        validation_data=val_dataset,
        validation_steps=val_steps
    )
    # Reset the original generators
    train_generator.reset()
    val_generator.reset()
    return history

def fine_tune_model(model, base_model, train_dataset, val_dataset, train_generator, val_generator, config):
    for layer in base_model.layers[-5:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=config['model']['fine_tune_learning_rate'] * 0.5),
                loss='binary_crossentropy', metrics=['accuracy'])
    steps_per_epoch = train_generator.samples // config['data']['batch_size']
    val_steps = val_generator.samples // config['data']['batch_size']
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config['model']['fine_tune_epochs'],
        validation_data=val_dataset,
        validation_steps=val_steps
    )
    train_generator.reset()
    val_generator.reset()
    return history