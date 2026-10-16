"""Symulacja toru QPSK (zamiennik overlayu `rfsoc_qpsk` do pracy bez płytki RFSoC).

Tor: symbole QPSK -> filtr kształtujący RRC (SPS próbek/symbol) -> kanał (szum, offset
częstotliwości i fazy, opóźnienie ułamkowe) -> dec -> csync -> rrc -> tsync.
Interfejs `tx` / `rx` odpowiada sterownikom z notebooka (metody get_*, set_gain).
"""
import numpy as np

SPS = 8              # próbek na symbol
BETA = 0.35          # współczynnik roll-off RRC
SPAN = 8             # długość filtru w symbolach
RAMKA_SYM = 16       # symboli w ramce nadajnika (16 * SPS = 128 próbek, jak w overlayu)
RAMKA_RX = 64        # symboli w ramce odbiornika


def rrc(beta=BETA, sps=SPS, span=SPAN):
    """Filtr pierwiastek z podniesionego cosinusa, znormalizowany do energii 1."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                   / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def _opoznienie_ulamkowe(x, d):
    """Opóźnienie o d próbek (0<=d<1) interpolacją liniową."""
    return (1 - d) * x + d * np.concatenate(([x[0]], x[:-1]))


class QpskSim:
    def __init__(self, sigma=0.05, cfo=2e-4, faza=0.6, opoznienie=0.37, seed=None):
        """sigma: odch. std. szumu ADC (stałe, niezależne od gain); cfo: offset częstotliwości
        w cyklach/próbkę; faza: offset fazy [rad]; opoznienie: opóźnienie ułamkowe [próbki]."""
        self.sigma, self.cfo, self.faza, self.opoznienie = sigma, cfo, faza, opoznienie
        self.gain = 1.0
        self.rng = np.random.default_rng(seed)
        self.h = rrc()

    # --- nadajnik ---
    def set_gain(self, g):
        self.gain = float(g)

    def _symbole(self, n):
        k = self.rng.integers(0, 4, n)
        return np.exp(1j * (np.pi / 4 + k * np.pi / 2))      # punkty (±1 ±1j)/√2

    def _ksztaltuj(self, s):
        up = np.zeros(len(s) * SPS, complex)
        up[::SPS] = s
        return np.convolve(up, self.h)[len(self.h) // 2: len(self.h) // 2 + len(up)] * self.gain

    def get_many_symbols(self, n_ramek=1):
        return self._symbole(n_ramek * RAMKA_SYM)

    def get_symbols(self):
        return self._symbole(RAMKA_SYM)

    def get_shaped_time(self):
        return self._ksztaltuj(self._symbole(RAMKA_SYM + SPAN))[SPAN // 2 * SPS: (SPAN // 2 + RAMKA_SYM) * SPS]

    # --- odbiornik: cały tor dla n_sym symboli, zwraca słownik etapów ---
    def tor(self, n_sym):
        s = self._symbole(n_sym)
        x = self._ksztaltuj(s)
        n = np.arange(len(x))
        x = _opoznienie_ulamkowe(x, self.opoznienie)
        x = x * np.exp(1j * (2 * np.pi * self.cfo * n + self.faza))
        x = x + self.sigma * (self.rng.standard_normal(len(x)) + 1j * self.rng.standard_normal(len(x))) / np.sqrt(2)
        dec = x

        # csync: estymacja CFO z widma x^4 (pik na 4*cfo), korekcja częstotliwości
        N = len(dec)
        okno = np.hanning(N)
        P = np.abs(np.fft.fft(dec ** 4 * okno, 8 * N))
        f = np.fft.fftfreq(8 * N)
        cfo_est = f[np.argmax(P)] / 4
        csync = dec * np.exp(-2j * np.pi * cfo_est * n)

        rrc_out = np.convolve(csync, self.h)[len(self.h) // 2: len(self.h) // 2 + N]

        # tsync: wybór fazy próbkowania o największej średniej energii, potem co SPS próbek
        off = np.argmax([np.mean(np.abs(rrc_out[o::SPS]) ** 2) for o in range(SPS)])
        tsync = rrc_out[off::SPS]
        tsync = tsync[SPAN // 2: -SPAN // 2]          # odcięcie stanów przejściowych filtrów
        return {"dec": dec, "csync": csync, "rrc": rrc_out, "tsync": tsync}

    def _ramka(self, etap, n):
        d = self.tor(RAMKA_RX + 2 * SPAN)[etap]
        return d[SPAN: SPAN + n] if etap == "tsync" else d[SPAN * SPS: SPAN * SPS + n]

    def get_decimated(self):      return self._ramka("dec", RAMKA_RX * SPS)
    def get_coarse_synced(self):  return self._ramka("csync", RAMKA_RX * SPS)
    def get_rrced(self):          return self._ramka("rrc", RAMKA_RX * SPS)
    def get_data(self):           return self._ramka("tsync", RAMKA_RX)


def ewm(s):
    """EVM [%] względem najbliższych punktów idealnej konstelacji QPSK po derotacji.
    Orientacja z fazy średniej s^4 (konstelacja QPSK ma 4 symetrię)."""
    s = np.asarray(s)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    obrot = np.angle(np.mean(s ** 4)) / 4 - np.pi / 4      # ideał leży na π/4 + kπ/2
    s = s * np.exp(-1j * obrot)
    ideal = (np.sign(s.real) + 1j * np.sign(s.imag)) / np.sqrt(2)
    return 100 * np.sqrt(np.mean(np.abs(s - ideal) ** 2) / np.mean(np.abs(ideal) ** 2))
