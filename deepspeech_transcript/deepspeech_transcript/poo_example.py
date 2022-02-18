import numpy as np


class Animal(object):
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def eat(self, food: str) -> str:
        if food == "vegetables":
            return f"{food} are delicious!"
        else:
            return f"I dont like {food}"

    def is_adult(self) -> bool:
        return self.age > 18


class Dog(Animal):
    def walk(self, steps: int) -> str:
        return f"I have walked {steps} steps."


class Cat(Animal):
    def walk(self, steps: int) -> str:
        return f"I have walked {steps-1} steps."


bobby = Dog("Bobby", 3)
print(bobby.eat("meal"))
print(bobby.is_adult())
print(bobby.walk(10))
carla = Cat("Carla", 10)
print(carla.eat("meal"))
print(carla.is_adult())
print(carla.walk(10))

print(bobby.age)
