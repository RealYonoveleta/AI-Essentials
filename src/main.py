from ai_essentials.value import Value


x = Value(2.0)
y = Value(3.0)
z = x * y
out = z + 2.0
out.backward()

print(x.grad)  # should be 3.0
print(y.grad)  # should be 2.0
