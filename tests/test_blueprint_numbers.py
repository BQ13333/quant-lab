from math import sqrt, exp, log, erf, pi
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
near = lambda a,b,tol: abs(a-b) <= tol
def arl(k, h, d):
    D = d - k; b = h + 1.166
    return (exp(-2 * D * b) + 2 * D * b - 1) / (2 * D * D)

def test_blueprint_arithmetic():
    A = [arl(0.15, hv, sh) for hv in (3, 5) for sh in (0, 0.3)]
    assert all(near(u, v, 0.15) for u, v in zip(A, [27.6, 11.9, 78.0, 22.4]))
    assert near(arl(0.25, 7, 0), 433.9, 1.0) and near(arl(0.25, 7, 0.5), 24.8, 0.2)       # rejected design
    # 3.1 horizon half-life: daily alpha rates in bp per day per 1 sd of signal
    x = [1, 3.5, 8, 15.5, 30.5, 50.5]; aa = [2.224, 1.853, 1.657, 1.068, 0.589, 0.240]
    assert all(v > 0 for v in aa)                                                         # else use the NLS route
    y = [log(v) for v in aa]; xm, ym = sum(x) / 6, sum(y) / 6
    slope = sum((p - xm) * (s - ym) for p, s in zip(x, y)) / sum((p - xm) ** 2 for p in x)
    a0 = exp(ym - slope * xm)
    assert near(-slope, 0.0444, 0.0003) and near(log(2) / -slope, 15.6, 0.15) and near(a0, 2.25, 0.03)
    assert near(sum(a0 * exp(slope * dy) for dy in range(1, 22)), 30.1, 0.4)              # = IC x sigma_xs = 30 bp
    assert near(((1.645 + 0.842) / 0.5) ** 2, 24.7, 0.1) and near(((1.645 + 0.842) / 1.039) ** 2, 5.7, 0.05)
    # 3.6 drawdown: stationary (IR > 0 only) and zero-drift first passage
    assert near(exp(-2 * 0.5 * 2), 0.135, 0.001) and near(exp(-2 * 0.5 * 3), 0.050, 0.001)
    fp = lambda dv, T: 1 - sum((-1) ** jj * 4 / ((2 * jj + 1) * pi)
                               * exp(-((2 * jj + 1) ** 2) * pi ** 2 * T / (8 * dv * dv)) for jj in range(60))
    assert near(fp(2, 1), 0.091, 0.002) and near(fp(2, 3), 0.495, 0.003)
    assert near(fp(3, 1), 0.0054, 0.0005) and near(fp(3, 3), 0.1665, 0.003)
    # 3.7 cost-change sensitivity; 6.8 pilot design, funding and reserve; ramp size increment
    assert near(2 * 0.0005 * 0.60 / 0.14, 0.0043, 0.0001)
    sd_day = 0.5 * sqrt((1 + 0.3) / 2); infl = 1 + 2 * (2 / 3) * 0.2
    se30, se24 = sd_day * sqrt(infl / 30), sd_day * sqrt(infl / 24)
    assert near(sd_day, 0.403, 0.001) and near(infl, 1.267, 0.001)
    assert near(1.717 * se30, 0.142, 0.002) and 1.717 * se30 <= 0.15 and 1.740 * se24 > 0.15
    assert near(Phi((0.35 - 1.717 * se30 - 0.20) / se30), 0.54, 0.02)
    fund = [6 * 1.25 * tk * 1.10 * fxr for tk, fxr in ((5000, 1), (15000, 0.55), (15000, 0.60))]
    assert all(near(u, v, 0.01) for u, v in zip(fund, [41250, 68062.5, 74250]))
    assert near(sum(fund), 183562.5, 0.01) and fund[2] <= min(75000, 0.20 * 375000)
    assert near(30 * 6250 * 0.0098 + 6 * 6250 * 0.0049, 2021.25, 0.01) and 2021.25 < 3750
    assert near(30 * 18750 * 0.0094 + 6 * 18750 * 0.0047, 5816.25, 0.01) and 5816.25 < 11250
    assert near(30 * 18750 * 0.00346 + 6 * 18750 * 0.00148, 2112.75, 0.01) and 2112.75 < 11250
    I = lambda qd: 0.02 * sqrt(qd / 3e6) * 100                                            # Bursa unit impact, %
    assert near(I(9900), 0.115, 0.001) and near(2 * (I(9900) - I(4950)), 0.067, 0.002)
    # 3.4 / 5.1 capacity and impact; 5.2-5.4 event impact (x2 event multiplier); 5.5 constrained HK sleeve
    assert near(0.04 * sqrt(0.6 / 180) * 100, 0.23, 0.005) and near(0.0625 * 180, 11.25, 1e-9)
    ev = lambda T, s, Pn, D, adv: 2 * 2 * T * s * sqrt(Pn / (D * adv)) * 100
    assert near(ev(2, 0.020, 6250, 3, 5e6), 0.33, 0.01) and near(ev(3, 0.023, 18750, 3, 150e6), 0.18, 0.01)
    assert near(ev(2, 0.025, 18750, 5, 20e6), 0.27, 0.01) and near(ev(2, 0.030, 6250, 5, 0.5e6), 1.20, 0.01)
    assert near(2 * 1.0 * 0.022 * sqrt(47700 / (3 * 60e6)) * 100, 0.072, 0.001)
    assert near(20 * 3 * 60 * (0.02925 / (8 * 0.022)) ** 2, 99.4, 0.5) and near(11 * 1.75 + 25, 44.25, 1e-9)
    # 1.4 and 7.1 figures
    assert near(525000 * 0.035, 18375, 1e-6) and near(sqrt(0.005 * 0.995 / 50000) * 100, 0.0315, 0.0005)
    assert near(sqrt(0.00125 * 0.99875 / 1e6) * 100, 0.0035, 0.0002)
    assert near(Phi(1.08 - 2.807), 0.042, 0.002) and near(Phi(1.80 - 2.807), 0.157, 0.003)
    print("all assertions passed")
