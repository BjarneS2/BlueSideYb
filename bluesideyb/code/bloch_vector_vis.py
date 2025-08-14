if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from IonQubit import IonQubit
    qubit = IonQubit(initial_state='+')  # you can also use 0

    # check Bloch vector
    print("Bloch vector:", qubit.bloch_vector())

    # plot the qubit state
    qubit.plot_vector_matplotlib(show_arrow=True)