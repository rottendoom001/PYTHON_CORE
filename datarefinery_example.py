from datarefinery.tuple.TupleOperations import wrap, keep, substitution
from datarefinery.Tr import Tr

x2 = wrap(lambda x: x*2)

tr = Tr(keep(["name"])).then(substitution(["value"], x2))
operation = tr.apply()
(inp, res, err) = operation({"name": "John", "value": 10})
print(res)
