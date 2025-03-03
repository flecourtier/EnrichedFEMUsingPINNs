import dolfin as df
import mshr

mesh = df.RectangleMesh(df.Point(0, 0), df.Point(1, 1), 10, 10)

print("mesh created")
print("It worked!")
