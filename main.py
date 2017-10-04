from math import sin, pi, ceil, sqrt, cos
import matplotlib.pyplot as plt

INITIAL_PHASE = pi / 8


def harmonic_signal_function(n, N, phase=0.0):
    return sin(2 * pi * n / N + phase)


def calc_M_vector(N):
    K = 3 * N / 4
    K = 0

    MAX_M = N * 5

    M_vector = [N - 1]

    step = (MAX_M - K) / (4 * 8)
    m = N - 1 - step
    while m > K:
        M_vector.insert(0, round(m))
        m -= step

    m = N - 1 + step
    while m < MAX_M:
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


def calc_amplitude(signal, M):
    pack = zip(signal, [i for i in range(len(signal))])
    Asin = (2 / M) * sum(map(lambda a: a[0] * sin(2 * pi * a[1] / M), pack))
    Acos = (2 / M) * sum(map(lambda a: a[0] * cos(2 * pi * a[1] / M), pack))
    return sqrt(Asin * Asin + Acos * Acos)


def calc_amplitude_vector(signals, M_vector):
    pack = zip(signals, M_vector)
    return [calc_amplitude(signal, m) for signal, m in pack]


def main():
    N = 128
    M_vector = calc_M_vector(N)

    # without initial phase
    signals = [[harmonic_signal_function(n, N) for n in range(0, m + 1)] for m in M_vector]
    # with initial phase
    signals_with_phase = [[harmonic_signal_function(n, N, INITIAL_PHASE) for n in range(0, m + 1)] for m in M_vector]

    # mean_square_value_vector_with_phase = [calc_mean_square_value(signal) for signal in signals]
    mean_square_value_vector = [calc_mean_square_value(signal) for signal in signals_with_phase]
    math_exp_vector = [calc_math_expectation(signal) for signal in signals]
    pack = zip(signals, math_exp_vector)
    mean_square_deviation_vector = [calc_mean_square_deviation(a[0], a[1]) for a in pack]
    # mean_square_value_error_vector_with_phase = [calc_mean_square_value_error(msv) for msv in mean_square_value_vector_with_phase]
    mean_square_value_error_vector = [calc_mean_square_value_error(msv) for msv in mean_square_value_vector]
    mean_square_deviation_error_vector = [calc_mean_square_value_error(msd) for msd in mean_square_deviation_vector]
    # plt.plot(M_vector, mean_square_value_error_vector)
    # plt.show()

    # plt.plot(M_vector, calc_amplitude_vector(signals, M_vector))
    # plt.plot(M_vector, list(map(lambda a: calc_amplitude_error(a), calc_amplitude_vector(signals, M_vector))))
    # plt.show()

    plt.figure(1)
    plt.subplot(211)
    plt.title('Ошибка СКЗ')
    plt.plot(M_vector, mean_square_value_error_vector)
    plt.plot(M_vector, mean_square_deviation_error_vector)
    # plt.plot(M_vector, mean_square_value_error_vector_with_phase)

    plt.subplot(212)
    plt.title('Ошибка амплитуды')
    # plt.plot(M_vector, calc_amplitude_vector(signals_with_phase, M_vector))
    # plt.plot(M_vector, calc_amplitude_vector(signals, M_vector))
    # plt.plot(M_vector, list(map(lambda a: calc_amplitude_error(a), calc_amplitude_vector(signals_with_phase, M_vector))))
    plt.plot(M_vector, list(map(lambda a: calc_amplitude_error(a), calc_amplitude_vector(signals, M_vector))))

    plt.show()


if __name__ == "__main__":
    main()