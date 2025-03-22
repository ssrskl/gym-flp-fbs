from loguru import logger
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self):
        return f"{self.name} is {self.age} years old."

# 创建一个Person对象
person = Person("猫颜", 30)
# 打印对象的字符串表示
logger.info(person)
