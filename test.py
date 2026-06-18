def add_layer_BROKEN(layers=[]):
    layers.append("linear")
    return layers

print("BROKEN version:")
print(f"  Call 1: {add_layer_BROKEN()}")  # ['linear']
print(f"  Call 2: {add_layer_BROKEN()}")  # ['linear', 'linear'] ← BUG!