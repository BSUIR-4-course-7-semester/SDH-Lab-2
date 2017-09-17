from math import sin, pi, ceil, sqrt, cos
import matplotlib.pyplot as plt


def harmonic_signal_function(n, N, phase=0):
    return sin(2 * pi * n / N + phase)


def calc_M_vector(N):
    K = 3 * N / 4
    twice_N = N * 2

    M_vector = [N - 1]

    step = (twice_N - K) / (4 * 8)
    m = N - 1 - step
    while m > K:
        M_vector.insert(0, round(m))
        m -= step

    m = N - 1 + step
    while m < twice_N:
        M_vector.append(round(m))
        m += step

    return M_vector


def calc_mean_square_value(vector):
    return sqrt(sum([v * v for v in vector]) / len(vector))

NORMAL_MSV = 0.707


def calc_mean_square_value_error(mean_square_value):
    return NORMAL_MSV - mean_square_value

NORMAL_AMPLITUDE = 1


def calc_amplitude_error(amplitude):
    return NORMAL_AMPLITUDE - amplitude


def calc_mean_square_deviation(vector, mathematical_expectation):
    return sqrt(sum([v * v - pow(mathematical_expectation, 2) for v in vector]) / len(vector))


def calc_math_expectation(vector):
    return sum(vector) / len(vector)


# def calc_a_vector_for_signal(signal, T):
#     pack = zip(signal, [i for i in range(T)])
#     return [2 / T * val * sin(2 * pi * n) for val, n in pack]
#
#
# def calc_b_vector_for_signal(signal, T):
#     pack = zip(signal, [i for i in range(T)])
#     return [2 / T * val * (1 - cos(2 * pi * n)) for val, n in pack]
#
#
# def calc_amplitude(a_vector, b_vector):
#     pack = zip(a_vector, b_vector)
#     return max([sqrt(val_a * val_a + val_b * val_b) for val_a, val_b in pack]) * pow(2, 0.5)


def main():
    N = 64
    M_vector = calc_M_vector(N)

    signals = [[harmonic_signal_function(n, N) for n in range(0, m + 1)] for m in M_vector]
    mean_square_value_vector = [calc_mean_square_value(signal) for signal in signals]
    mean_square_deviation_vector = [calc_mean_square_deviation(signal, calc_math_expectation(signal)) for signal in signals]

    mean_square_value_error_vector = [calc_mean_square_value_error(msv) for msv in mean_square_value_vector]
    mean_square_deviation_error_vector = [calc_mean_square_value_error(msd) for msd in mean_square_deviation_vector]

    # pack = zip(signals, M_vector)
    # amplitude_vector = [
    #     calc_amplitude(
    #         calc_a_vector_for_signal(signal, m),
    #         calc_b_vector_for_signal(signal, m)
    #     ) for signal, m in pack
    # ]

    # amplitude_error_vector = [calc_amplitude_error(val) for val in amplitude_vector]

    plt.figure(1)
    plt.subplot(211)
    plt.title('Ошибка СКЗ')
    plt.plot(mean_square_value_error_vector)

    plt.subplot(212)
    plt.title('Ошибка СК отклонения')
    plt.plot(mean_square_deviation_error_vector)

    # plt.subplot(222)
    # plt.title('Амплитуда')
    # plt.plot(amplitude_error_vector)
    plt.show()


if __name__ == "__main__":
    main()