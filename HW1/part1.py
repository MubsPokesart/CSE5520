import pygal

def fibonacci(Nterms: int, N1: int, N2: int):
    Count = 0
    # check if the number of terms is valid
    if Nterms <= 0:
        print("Please enter a positive integer")

    # if there is only one term, return n1
    elif Nterms == 1:
        print("Fibonacci sequence upto",Nterms,": ")
        print(N1)
    
    # generate fibonacci sequence
    else:
        print("Fibonacci sequence:")
        while Count < Nterms:
            print(N1)
            Nth = N1 + N2
            # update values
            N1 = N2
            N2 = Nth
            Count += 1
    return N1


def generate_plot_items(num:int):
    fib_items = []
    for i in range(1, num + 1):
        fib_items.append(fibonacci(i, 0, 1))

    bar_chart = pygal.Bar()
    bar_chart._title = f'Fixed Fibonacci Sequence (First {num} terms)'
    bar_chart.x_labels = list(range(1, num + 1))
    bar_chart.add('Fibonacci', fib_items)
    
    return bar_chart

if __name__ == "__main__":
    generate_plot_items(10).render_to_file('fibonacci_10_terms.svg')
