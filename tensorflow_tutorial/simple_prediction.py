from dataclasses import dataclass
from enum import Enum

import numpy as np
import tensorflow as tf


class Category(Enum):
    PREMIUM = 0
    MEDIUM = 1
    BASIC = 2


@dataclass
class Person:
    name: str
    age: int
    city: str
    favorite_color: str


__normalized_cities = dict()
__normalized_colors = dict()
__normalized_categories = dict()
__max_age = 0
__min_age = 0


def normalize_string_set(data: set[str], output: dict[str, list[float]]):
    n = len(data)
    for index, val in enumerate(data):
        output[val] = [0] * n
        output[val][index] = 1


def normalize_cities(input_data: list[tuple[Person, Category]]):
    cities_set = {item[0].city for item in input_data}
    normalize_string_set(cities_set, __normalized_cities)


def normalize_colors(input_data: list[tuple[Person, Category]]):
    colors_set = {item[0].favorite_color for item in input_data}
    normalize_string_set(colors_set, __normalized_colors)


def normalize_age(age: int):
    return (age - __min_age) / (__max_age - __min_age)


def normalize_person(person: Person):
    normalized_person = [normalize_age(person.age)]
    normalized_person += __normalized_cities[person.city]
    normalized_person += __normalized_colors[person.favorite_color]
    return np.array(normalized_person)


def normalize_input(input_data: list[tuple[Person, Category]]):
    people = []
    categories = []
    for item in input_data:
        normalized_person = normalize_person(item[0])
        people.append(normalized_person)
        categories.append(item[1].value)
    return np.array(people), np.array(categories)


def train_model(x, y) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=[7], units=80, activation='relu'))
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=100, shuffle=True)
    return model


def predict(person: Person, model: tf.keras.Sequential) -> Category:
    normalized_person = normalize_person(person)
    prediction = model.predict(np.array([normalized_person]))
    return Category(prediction.argmax())


def main():
    global __max_age
    global __min_age
    p1 = Person(name='Erick', age=30, city='São Paulo', favorite_color='blue')
    p2 = Person(name='Ana', age=25, city='Rio de Janeiro', favorite_color='red')
    p3 = Person(name='Carlos', age=40, city='Curitiba', favorite_color='green')
    training_input = [
        (p1, Category.PREMIUM),
        (p2, Category.MEDIUM),
        (p3, Category.BASIC)
    ]
    ages_set = {item[0].age for item in training_input}
    __max_age = max(ages_set)
    __min_age = min(ages_set)
    normalize_cities(training_input)
    normalize_colors(training_input)
    x_train, y_train = normalize_input(training_input)
    model = train_model(x_train, y_train)
    test_person = Person(name='Zé', age=28, city='São Paulo', favorite_color='blue')
    category = predict(test_person, model)
    print(f'The category of {test_person.name} is {category.name}.')


if __name__ == "__main__":
    main()
