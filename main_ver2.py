import os

# 기본 경로
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

base_dir = '/Users/younglimmm/PycharmProjects/FindingIdealAI/imageDirectory' # 이게 맞나?

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 훈련에 사용되는 고양이/강아지/공룡/토끼상 이미지 경로

train_cats_dir = os.path.join(train_dir, 'cat')
train_dogs_dir = os.path.join(train_dir, 'dog')
train_dinosaur_dir = os.path.join(train_dir, 'dinosaur')
train_rabbit_dir = os.path.join(train_dir, 'rabbit')
print(train_cats_dir)
print(train_dogs_dir)
print(train_dinosaur_dir)
print(train_rabbit_dir)

# 테스트에 사용되는 고양이/강아지/공룡/토끼상 이미지 경로

validation_cats_dir = os.path.join(validation_dir, 'cat')
validation_dogs_dir = os.path.join(validation_dir, 'dog')
validation_dinosaur_dir = os.path.join(validation_dir, 'dinosaur')
validation_rabbit_dir = os.path.join(validation_dir, 'rabbit')
print(validation_cats_dir)
print(validation_dogs_dir)
print(validation_dinosaur_dir)
print(validation_rabbit_dir)

print('Total training cat images :', len(os.listdir(train_cats_dir)))
print('Total training dog images :', len(os.listdir(train_dogs_dir)))
print('Total training dinosaur images :', len(os.listdir(train_dinosaur_dir)))
print('Total training rabbit images :', len(os.listdir(train_rabbit_dir)))

print('Total validation cat images :', len(os.listdir(validation_cats_dir)))
print('Total validation dog images :', len(os.listdir(validation_dogs_dir)))
print('Total validation dinosaur images :', len(os.listdir(validation_dinosaur_dir)))
print('Total validation rabbit images :', len(os.listdir(validation_rabbit_dir)))

# 모델링
import tensorflow as tf

model = Sequential()


model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation="softmax"))

model.summary()


# 모델 컴파일링


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 전처리


train_datagen = ImageDataGenerator(rescale = 1.0/255. )
test_datagen = ImageDataGenerator(rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode = 'categorical',
                                                    target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='categorical',
                                                        target_size=(150, 150))

print(train_generator.class_indices)
print(validation_generator.class_indices)

# 훈련

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=200,
                    epochs=20,
                    validation_steps=50,
                    verbose=1)

model.save('./ver2_new_model')






