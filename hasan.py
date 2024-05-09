import math

USERNAME = "hasan"
PASSWORD = "hasan@123"

def get_username():
    username = input("Enter your UserName: ")
    return username == USERNAME

def get_password():
    attempts = 0
    while attempts < 3:
        password = input("Enter your Password: ")
        if password == PASSWORD:
            return True
        else:
            attempts += 1
            print("Incorrect password please enter again")
    print("Incorrect Password!! Sorry :(")
    return False

def calculator():
    while True:
        operator = input("Enter an operator: ").strip()
        if operator in ['+', '-', '*', '/', '^', 'sq']:
            x = float(input("Enter first operand: "))
            if operator in ['+', '-', '*', '/', '^']:
                y = float(input("Enter second operand: "))

        elif operator in ['abs', 'neg']:
            x = float(input("Enter a number: "))

        if operator == '+':
            result = x + y
            print(f"{x} + {y} = {result}")
        elif operator == '-':
            result = x - y
            print(f"{x} - {y} = {result}")
        elif operator == '*':
            result = x * y
            print(f"{x} * {y} = {result}")
        elif operator == '/':
            if y == 0:
                print("Error: Division by zero")
            else:
                result = x / y
                print(f"{x} / {y} = {result}")
        elif operator == '^':
            result = pow(x, y)
            print(f"{x} ^ {y} = {result}")
        elif operator == 'sq':
            if x < 0:
                print("Error: Square root of negative number")
            else:
                result = math.sqrt(x)
                print(f"sq ({x}) = {result}")
        elif operator == 'abs':
            result = abs(x)
            print(f"| {x} | = {result}")
        elif operator == 'neg':
            result = -x
            print(f"negation of {x} = {result}")
        else:
            print("Invalid operation")

        if input("Would you like to continue: ").lower() != 'yes':
            break

def main():
    if get_username():
        if get_password():
            print("Welcome to the calculator app!")
            calculator()
        else:
            print("Access Denied.")
    else:
        print("Invalid username")


main()
