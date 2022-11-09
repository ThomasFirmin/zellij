# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:39:05+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import random
import time
import enlighten


def best_counter(manager):
    return manager.counter(desc="Improvements", unit="solutions", leave=False)


def calls_counter(manager, total):

    bar_format = (
        "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "
        + "Pending:{count_0:{len_total}d} "
        + "Explor:{count_2:{len_total}d} "
        + "Exploi:{count_1:{len_total}d}"
    )

    pending = manager.counter(
        total=total,
        desc="Loss calls",
        unit="calls",
        color="white",
        bar_format=bar_format,
        leave=False,
    )
    exploitation = pending.add_subcounter("orange")
    exploration = pending.add_subcounter("cyan")

    return exploration, exploitation, pending


def metaheuristic_counter(manager, total, name):
    bar_format = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "

    counter = manager.counter(
        total=total,
        desc=f"   {name}",
        unit="calls",
        color="white",
        bar_format=bar_format,
        leave=False,
    )
    return counter


def calls_counter_inside(manager, total):
    bar_format = "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "

    counter = manager.counter(
        total=total,
        desc=f"      Evaluating",
        unit="calls",
        color="white",
        bar_format=bar_format,
        leave=False,
    )
    return counter


def best_found(manager, score):
    status_bar = manager.status_bar(
        "      Current score:{:.3f} | Best score:{:.3f}".format(score, score),
        color="white",
        leave=False,
    )
    return status_bar
