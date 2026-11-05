
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-03 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Запуск: python lab03.py --variant N [--quick] [--no-plots]

Один файл делает всё: self_check, демонстрацию стабильности, замеры, таблицу
счётчиков и графики (lab03_loglog_by_alg.png, lab03_random_all.png).
--quick — уменьшенные размеры для быстрой проверки (результаты для отчёта
брать из полного запуска).
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list, key=None) -> tuple[list, int, int]:
    key = key or (lambda x: x)
    res = list(a)
    n = len(res)
    comparisons = swaps = 0
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            comparisons += 1
            if key(res[j]) > key(res[j + 1]):
                res[j], res[j + 1] = res[j + 1], res[j]
                swaps += 1
                swapped = True
        if not swapped:
            break
    return res, comparisons, swaps


def insertion_sort(a: list, key=None) -> tuple[list, int, int]:
    key = key or (lambda x: x)
    res = list(a)
    comparisons = moves = 0
    for i in range(1, len(res)):
        x = res[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if key(res[j]) > key(x):
                res[j + 1] = res[j]
                moves += 1
                j -= 1
            else:
                break
        res[j + 1] = x
    return res, comparisons, moves


def selection_sort(a: list, key=None) -> tuple[list, int, int]:
    key = key or (lambda x: x)
    res = list(a)
    n = len(res)
    comparisons = swaps = 0
    for i in range(n - 1):
        m = i
        for j in range(i + 1, n):
            comparisons += 1
            if key(res[j]) < key(res[m]):
                m = j
        if m != i:
            res[i], res[m] = res[m], res[i]
            swaps += 1
    return res, comparisons, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None, key=None) -> list:

    if rng is None:
        rng = random.Random()
    key = key or (lambda x: x)
    res = list(a)
    _quick(res, 0, len(res), rng, key)
    return res


def _quick(res: list, lo: int, hi: int, rng: random.Random, key) -> None:
    while hi - lo > 1:
        pivot = key(res[rng.randrange(lo, hi)])
        lt, i, gt = lo, lo, hi
        while i < gt:
            k = key(res[i])
            if k < pivot:
                res[lt], res[i] = res[i], res[lt]
                lt += 1
                i += 1
            elif k > pivot:
                gt -= 1
                res[i], res[gt] = res[gt], res[i]
            else:
                i += 1

        if lt - lo < hi - gt:
            _quick(res, lo, lt, rng, key)
            lo = gt
        else:
            _quick(res, gt, hi, rng, key)
            hi = lt


# ---------------------------------------------------------------------------
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------


def stability_demo() -> None:
    """Показать, какие из четырёх сортировок стабильны (сравниваем только p[0])."""
    sorts = {
        "bubble_sort": lambda a: bubble_sort(a, key=lambda p: p[0])[0],
        "insertion_sort": lambda a: insertion_sort(a, key=lambda p: p[0])[0],
        "selection_sort": lambda a: selection_sort(a, key=lambda p: p[0])[0],
        "quick_sort": lambda a: quick_sort(a, random.Random(0), key=lambda p: p[0]),
    }
    arrays = [
        [(2, "a"), (1, "b"), (2, "c"), (1, "d")],
        [(2, "a"), (2, "b"), (1, "c")],
        [(3, "a"), (1, "b"), (3, "c"), (2, "d"), (1, "e"), (3, "f"), (2, "g"), (1, "h")],
    ]
    print("\nstability_demo:")
    for arr in arrays:
        print(f"  вход: {arr}")
        for name, fn in sorts.items():
            res = fn(arr)

            stable = res == sorted(arr, key=lambda p: p[0])
            print(f"    {name:<15} {res}  {'стабильна' if stable else 'НЕ стабильна'}")


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def check_algorithm(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],
        [7],
        [5, 5, 5, 5],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(is_sorted(res), f"нарушен порядок на {a}: получено {res}")
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")


    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        res = fn(list(a))
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_counters() -> None:
    """Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе."""
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    expect(comparisons == 10,
           f"selection_sort: неверный счётчик сравнений — {comparisons}, ожидалось 10")


def check_extra() -> None:
    """Собственные проверки: счётчики пузырька/вставок и стабильность."""
    asc, desc = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

    expect(bubble_sort(asc)[1:] == (4, 0), "bubble на упорядоченном: ожидалось (4, 0)")
    expect(insertion_sort(asc)[1:] == (4, 0), "insertion на упорядоченном: ожидалось (4, 0)")

    expect(bubble_sort(desc)[1:] == (10, 10), "bubble на обратном: ожидалось (10, 10)")
    expect(insertion_sort(desc)[1:] == (10, 10), "insertion на обратном: ожидалось (10, 10)")

    src = [3, 1, 2]
    for f in (bubble_sort, insertion_sort, selection_sort):
        f(src)
    quick_sort(src)
    expect(src == [3, 1, 2], "сортировка изменила входной список")

    pairs = [(2, "a"), (1, "b"), (2, "c"), (1, "d")]
    ref = sorted(pairs, key=lambda p: p[0])
    expect(bubble_sort(pairs, key=lambda p: p[0])[0] == ref, "bubble нестабилен")
    expect(insertion_sort(pairs, key=lambda p: p[0])[0] == ref, "insertion нестабилен")

    big = list(range(50_000))
    expect(quick_sort(big, random.Random(1)) == big, "quick_sort на 50000 упорядоченных")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""

    checks = [(name, lambda fn=fn: check_algorithm(fn)) for name, fn in ALGORITHMS.items()]
    checks.append(("счётчики", check_counters))
    checks.append(("доп. проверки", check_extra))
    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except NotImplementedError as err:
            where = traceback.extract_tb(err.__traceback__)[-1].name
            print(f"  {name}: не реализовано — {where} (TODO)")
            failed += 1
        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1
        except Exception:
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1
        else:
            print(f"  {name}: OK")
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; "
                         f"замеры выполняются только после всех проверок.")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5


def set_quick_mode() -> None:
    """Уменьшенные размеры и число повторов для быстрой проверки скрипта."""
    global SIZES_QUADRATIC, SIZES_QUICK, REPEATS
    SIZES_QUADRATIC = [100, 200, 400, 800, 1_600]
    SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 60_000]
    REPEATS = 3


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        arg = list(data)
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def counters_table(seed: int) -> None:
    """Таблица счётчиков (сравнения / обмены) для простых сортировок."""
    print("\nсчётчики (сравнения / обмены; для вставок — сдвиги):")
    for n in (1_000, 4_000):
        for cls, data in make_inputs(n, seed).items():
            for name, f in (("bubble_sort", bubble_sort), ("insertion_sort", insertion_sort),
                            ("selection_sort", selection_sort)):
                _, c, m = f(data)
                print(f"  n={n:>5} {cls:<13} {name:<15} сравнений={c:>10} обменов={m:>10}")


def run_benchmarks(seed: int) -> dict:
    """Замеры всех алгоритмов на трёх классах входов; возвращает словарь результатов."""
    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]
    results: dict = {}
    for name, fn, sizes in plans:
        print(f"\n{name}:", flush=True)
        for n in sizes:
            for cls, data in make_inputs(n, seed).items():
                t = bench(fn, data)
                results.setdefault(name, {}).setdefault(cls, {})[n] = t
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c", flush=True)
    counters_table(seed)
    return results


# ---------------------------------------------------------------------------
# 7. Графики log-log и наклоны
# ---------------------------------------------------------------------------


def make_plots(results: dict) -> None:
    """Строит log-log графики и печатает наклоны (МНК по log n, log t)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib/numpy не установлены: pip install matplotlib numpy — графики пропущены")
        return
    colors = {"упорядоченный": "tab:green", "случайный": "tab:blue", "обратный": "tab:red"}
    names = list(results)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    print("\nнаклоны log-log (теория: 2 для O(n^2), ~1.1-1.2 для n log n, 1 для O(n)):")
    for ax, name in zip(axes.flat, names):
        for cls, d in results[name].items():
            ns = np.array(sorted(d))
            ts = np.array([d[n] for n in ns])
            k = np.polyfit(np.log(ns), np.log(ts), 1)[0]
            print(f"  {name:<15}{cls:<14}{k:.3f}")
            ax.loglog(ns, ts, "o-", color=colors[cls], label=f"{cls} (наклон {k:.2f})")
        ax.set_title(name)
        ax.set_xlabel("n")
        ax.set_ylabel("время, с")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lab03_loglog_by_alg.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in names:
        d = results[name]["случайный"]
        ns = sorted(d)
        ax.loglog(ns, [d[n] for n in ns], "o-", label=name)
    ax.set_title("Случайный вход: все алгоритмы")
    ax.set_xlabel("n")
    ax.set_ylabel("время, с")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("lab03_random_all.png", dpi=130)
    print("\nграфики сохранены: lab03_loglog_by_alg.png, lab03_random_all.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--quick", action="store_true", help="уменьшенные размеры (для проверки)")
    ap.add_argument("--no-plots", action="store_true", help="не строить графики")
    args = ap.parse_args()
    if args.quick:
        set_quick_mode()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    stability_demo()
    results = run_benchmarks(seed)
    if not args.no_plots:
        make_plots(results)


if __name__ == "__main__":
    main()