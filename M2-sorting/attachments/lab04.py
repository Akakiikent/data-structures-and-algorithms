
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности.

Запуск: python lab04.py --variant N [--quick] [--no-plots]

Один файл делает всё: self_check, демонстрацию стабильности, устойчивость
внутреннего прохода RadixSort, влияние диапазона ключей на CountingSort, расход
памяти, бенчмарк против QuickSort (ЛР 3) и sorted(), графики (lab04_loglog.png).
--quick — уменьшенные размеры для быстрой проверки скрипта.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback
import tracemalloc

# ---------------------------------------------------------------------------
# 1. Сортировки (реализуются вручную; sorted() используется только как эталон)
# ---------------------------------------------------------------------------


def _ident(x):
    return x


def merge(left: list, right: list, key=_ident) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей элемент из left идёт раньше — это даёт
    нестрогое сравнение <=. Время слияния O(len(left) + len(right)).
    """
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def merge_sort(a: list, key=_ident) -> list:
    """MergeSort: возвращает новый отсортированный список (вход не меняется).

    Сложность: O(n log n) во всех случаях (глубина рекурсии log2 n, на каждом
    уровне слияния обрабатывается n элементов); память O(n); стабилен.
    """
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    return merge(merge_sort(a[:mid], key), merge_sort(a[mid:], key), key)


def counting_sort(a: list[int], key_max: int, key=_ident) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи — целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Время и память O(n + k), где k = key_max + 1. Ограничение: ключи целые,
    а диапазон k не должен быть много больше n — иначе массив счётчиков
    (и проход по нему) разрушает выигрыш.
    """
    count = [0] * (key_max + 1)
    for x in a:
        count[key(x)] += 1
    for v in range(1, key_max + 1):
        count[v] += count[v - 1]
    out = [None] * len(a)
    for x in reversed(a):
        count[key(x)] -= 1
        out[count[key(x)]] = x
    return out


def _stable_digit_pass(idx: list, ks: list, exp: int) -> list:
    """Устойчивая counting-сортировка индексов по цифре (ks[i] // exp) % 10."""
    count = [0] * 10
    for i in idx:
        count[ks[i] // exp % 10] += 1
    for d in range(1, 10):
        count[d] += count[d - 1]
    out = [0] * len(idx)
    for i in reversed(idx):
        d = ks[i] // exp % 10
        count[d] -= 1
        out[count[d]] = i
    return out


def _unstable_digit_pass(idx: list, ks: list, exp: int) -> list:
    """НЕустойчивый проход (выбором) — только для демонстрации, зачем нужна устойчивость."""
    idx = list(idx)
    for i in range(len(idx) - 1):
        m = i
        for j in range(i + 1, len(idx)):
            if ks[idx[j]] // exp % 10 < ks[idx[m]] // exp % 10:
                m = j
        idx[i], idx[m] = idx[m], idx[i]
    return idx


def radix_sort_lsd(a: list[int], key=_ident, digit_pass=_stable_digit_pass) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Цикл по разрядам (exp = 1, 10, 100, ...) от младшего к старшему; каждый
    проход — устойчивая counting-сортировка по цифре. Время O(d * (n + 10)),
    где d — число десятичных разрядов максимального ключа; память O(n).
    Устойчивость прохода обязательна: после сортировки по младшим разрядам
    элементы с равной текущей цифрой должны остаться упорядоченными по младшим.
    """
    if not a:
        return []
    ks = [key(x) for x in a]
    idx = list(range(len(a)))
    exp = 1
    mx = max(ks)
    while mx // exp > 0:
        idx = digit_pass(idx, ks, exp)
        exp *= 10
    return [a[i] for i in idx]


def sift_down(a: list, i: int, n: int, key=_ident) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте."""
    while True:
        left = 2 * i + 1
        if left >= n:
            return
        big = left
        if left + 1 < n and key(a[left + 1]) > key(a[left]):
            big = left + 1
        if key(a[big]) <= key(a[i]):
            return
        a[i], a[big] = a[big], a[i]
        i = big


def heap_sort(a: list, key=_ident) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Построение max-кучи просейками вниз снизу вверх — O(n); затем n раз:
    обмен вершины с последним элементом кучи и просейка — O(n log n).
    Итого O(n log n) всегда, память O(1). Нестабилен.
    """
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n, key)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end, key)


def quick_sort(a: list, rng: random.Random | None = None, key=_ident) -> list:
    """QuickSort с рандомизацией опорного элемента (из ЛР 3). Новый список.

    Средний O(n log n), худший O(n^2); трёхстороннее разбиение, рекурсия в
    меньшую часть — глубина стека не больше log2 n. Нестабилен.
    """
    if rng is None:
        rng = random.Random(0)
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
# 2. Верификация: инварианты + сверка с эталоном sorted()
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    """heap_sort сортирует на месте: проверяется тот список, который ему передали."""
    buf = list(a)
    heap_sort(buf)
    return buf



SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": lambda a: heap_sorted(a),
    "quick_sort": lambda a: quick_sort(a),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def check_sort(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],
        [7],
        [0],
        [5, 5, 5],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ]
    for a in boundary:
        src = list(a)
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(is_sorted(res), f"нарушен порядок на {a}: получено {res}")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")
        expect(a == src, "изменён входной список")

    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")
    for _ in range(20):
        a = [rng.randint(0, 999) for _ in range(rng.randint(500, 1500))]
        expect(fn(list(a)) == sorted(a), "расходится с sorted() на длинных входах")


def make_pairs(n: int, seed: int = 1) -> list[tuple[int, int]]:
    """Пары (ключ, порядковый номер) с большим числом повторов ключей."""
    rng = random.Random(seed)
    return [(rng.randint(0, 9), i) for i in range(n)]


def _by_key(p):
    return p[0]


STABILITY = {
    "merge_sort": lambda p: merge_sort(p, key=_by_key),
    "counting_sort": lambda p: counting_sort(p, key_max=9, key=_by_key),
    "radix_sort_lsd": lambda p: radix_sort_lsd(p, key=_by_key),
    "heap_sort": lambda p: _heap_copy(p),
    "quick_sort": lambda p: quick_sort(p, random.Random(0), key=_by_key),
}


def _heap_copy(p: list) -> list:
    buf = list(p)
    heap_sort(buf, key=_by_key)
    return buf


def is_stable(sort_fn, pairs: list) -> bool:
    """Эталон — sorted() по ключу (он стабилен): при равных ключах порядок номеров совпадёт."""
    return sort_fn(pairs) == sorted(pairs, key=_by_key)


def check_merge_stability() -> None:
    """Стабильность merge_sort на парах (ключ, порядковый номер)."""
    expect(is_stable(STABILITY["merge_sort"], make_pairs(100)),
           "merge_sort нестабилен: при равных ключах нарушен исходный порядок номеров")
    expect(merge([(1, "L")], [(1, "R")], key=_by_key) == [(1, "L"), (1, "R")],
           "слияние нарушило порядок равных ключей")


def check_linear_stability() -> None:
    """Counting и Radix заявлены стабильными — проверяем."""
    pairs = make_pairs(300)
    expect(is_stable(STABILITY["counting_sort"], pairs), "counting_sort нестабилен")
    expect(is_stable(STABILITY["radix_sort_lsd"], pairs), "radix_sort_lsd нестабилен")


def check_heap_property() -> None:
    """Свойство кучи после построения и сортировка на месте."""
    rng = random.Random(3)
    a = [rng.randint(0, 999) for _ in range(200)]
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    expect(all(a[(i - 1) // 2] >= a[i] for i in range(1, n)), "после построения нарушено свойство max-кучи")
    heap_sort(a)
    expect(is_sorted(a), "heap_sort не отсортировал список на месте")


def check_radix_and_counting_extra() -> None:
    """Большие ключи для radix, расстановка counting и неверность неустойчивого прохода."""
    rng = random.Random(4)
    big = [rng.randint(0, 10**9) for _ in range(20_000)]
    expect(radix_sort_lsd(big) == sorted(big), "radix_sort_lsd на 20000 чисел до 10^9")
    expect(counting_sort([3, 1, 3, 0], key_max=3) == [0, 1, 3, 3], "counting_sort на малом примере")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    checks.append(("стабильность counting/radix", check_linear_stability))
    checks.append(("свойство кучи / in-place", check_heap_property))
    checks.append(("radix/counting доп.", check_radix_and_counting_extra))
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
# 3. Демонстрации
# ---------------------------------------------------------------------------


def stability_demo() -> None:
    pairs = make_pairs(12, seed=3)
    print("\nstability_demo (пары (ключ, порядковый номер), сравнение по ключу):")
    print(f"  вход: {pairs}")
    for name, fn in STABILITY.items():
        print(f"  {name:<15} {'стабильна' if is_stable(fn, pairs) else 'НЕ стабильна'}")
    big = make_pairs(300)
    print("  на 300 парах:", ", ".join(f"{n}={'да' if is_stable(f, big) else 'нет'}" for n, f in STABILITY.items()))


def radix_stability_demo() -> None:
    """Показать, что с неустойчивым внутренним проходом RadixSort даёт неверный результат."""
    rng = random.Random(1)
    print("\nradix_stability_demo (внутренний проход: устойчивый vs неустойчивый):")
    for _ in range(1000):
        a = [rng.randint(10, 99) for _ in range(8)]
        bad = radix_sort_lsd(a, digit_pass=_unstable_digit_pass)
        if bad != sorted(a):
            print(f"  вход:                   {a}")
            print(f"  устойчивый проход:      {radix_sort_lsd(a)}  (верно)")
            print(f"  неустойчивый проход:    {bad}  (НЕВЕРНО)")
            return
    print("  (контрпример не найден)")


def counting_range_demo() -> list:
    """Как k влияет на время и память CountingSort при фиксированном n."""
    n = 10_000
    rng = random.Random(7)
    print(f"\ncounting_range_demo (n = {n}):")
    print(f"  {'k':>12} {'время, с':>12} {'память счётчиков':>18}")
    rows = []
    for k in (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        a = [rng.randrange(k) for _ in range(n)]
        t = bench_once(lambda x, k=k: counting_sort(x, key_max=k - 1), a)
        rows.append((k, t))
        print(f"  {k:>12} {t:>12.5f} {8 * k // 1024:>14} КиБ")
    print("  для k = 10^9 массив счётчиков занял бы ~8 ГБ — алгоритм теряет смысл (k >> n).")
    return rows


def memory_demo() -> None:
    """Пиковая память (tracemalloc, только списки Python), n = 100000, ключи 0..1000."""
    n = 100_000
    rng = random.Random(2)
    data = [rng.randint(0, 1000) for _ in range(n)]
    print(f"\nmemory_demo (n = {n}, ключи 0..1000; пик, КиБ; одна копия входа = {8 * n // 1024} КиБ):")
    algs = (("heap_sort", heap_sorted), ("quick_sort", quick_sort),
            ("counting_sort", lambda a: counting_sort(a, key_max=1000)),
            ("merge_sort", merge_sort), ("radix_sort_lsd", radix_sort_lsd))
    for name, fn in algs:
        arg = list(data)
        tracemalloc.start()
        fn(arg)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {name:<15} пик {peak // 1024:>6} КиБ")


# ---------------------------------------------------------------------------
# 4. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100
WIDE_KEY_MAX = 10**9


def set_quick_mode() -> None:
    global SIZES, REPEATS
    SIZES = [500, 1_500, 5_000, 15_000, 30_000]
    REPEATS = 3


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench_once(fn, data: list) -> float:
    buf = list(data)
    t0 = time.perf_counter()
    fn(buf)
    return time.perf_counter() - t0


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом."""
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> dict:
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort", quick_sort),
        ("sorted()", sorted),
    ]
    results: dict = {}
    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:", flush=True)
        for dist_name, data in datasets.items():
            parts = []
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":

                    parts.append(f"{algo_name}=пропуск")
                    continue
                t = bench(fn, data)
                results.setdefault(dist_name, {}).setdefault(algo_name, {})[n] = t
                parts.append(f"{algo_name}={t:.5f}")
            print(f"  {dist_name:<14} " + "  ".join(parts), flush=True)
    return results


# ---------------------------------------------------------------------------
# 5. Графики и наклоны
# ---------------------------------------------------------------------------


def make_plots(results: dict, counting_rows: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib/numpy не установлены: pip install matplotlib numpy — графики пропущены")
        return
    print("\nнаклоны log-log (теория: ~1.1-1.2 для n log n; ~1 для O(n + k) при малом k):")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (dist, algs) in zip(axes.flat, results.items()):
        for name, d in algs.items():
            ns = np.array(sorted(d))
            ts = np.array([d[n] for n in ns])
            k = np.polyfit(np.log(ns), np.log(ts), 1)[0]
            print(f"  {dist:<14}{name:<16}{k:.3f}")
            ax.loglog(ns, ts, "o-", label=f"{name} ({k:.2f})")
        ax.set_title(dist)
        ax.set_xlabel("n")
        ax.set_ylabel("время, с")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes.flat[-1]
    ax.loglog([k for k, _ in counting_rows], [t for _, t in counting_rows], "o-", color="tab:red")
    ax.set_title("CountingSort: время от k (n = 10000)")
    ax.set_xlabel("k = key_max + 1")
    ax.set_ylabel("время, с")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("lab04_loglog.png", dpi=120)
    print("\nграфик сохранён: lab04_loglog.png")


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
    radix_stability_demo()
    counting_rows = counting_range_demo()
    memory_demo()
    results = run_benchmarks(seed)
    if not args.no_plots:
        make_plots(results, counting_rows)


if __name__ == "__main__":
    main()