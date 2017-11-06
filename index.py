logeado = True
usuario = 'Alan'

def admin (f) :
    def comprobar (*args, **kwargs):
        if logeado :
            f (*args, **kwargs)
        else:
            print "No tiene permisos"
    return comprobar

def decorador (funcion) :
    def funcionDecoradora (*args, **kwargs):
        print "lalalala soy un decorador marica"
        funcion (*args, **kwargs)
    return funcionDecoradora

@admin
@decorador
def resta(n, m):
    print n-m

resta (5,3);
