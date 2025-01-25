#!/usr/bin/python3
from math import exp, floor, log, log10, ceil
from statistics import mean
import sys
from datetime import datetime, timedelta, date
import json
import threading
from time import time, sleep

CLEVELS_LIST = [('Easy', 1.0), ('Good', 5/6), ('Hard', 4/6),
                ('Lost', 1/2), ('Don\'t count [Selbststudium]', 0.0)]
CLEVELS_LIST.reverse()
CONFIDENCE_LEVELS = dict(CLEVELS_LIST)

BAR_LENGTH = 20


def print_help(program: str):
    print(f"Usage: {program} <path/to/file.csv> <option>")
    print("Options:")
    print("    help          Prints this message and exits")
    print("    add           Register study session in database")
    print("    history [n]   Print graph with amout of study for the last")
    print("                  n days (default n = 7).")
    print("    next          (default) Print next suggested study subject")
    print("    stats         Print some statistics")
    print("    record        Record study session duration")


def lev(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return lev(s2, s1)

    # Initialize the distance matrix
    distances = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        new_distances = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                new_distances.append(distances[j])
            else:
                new_distances.append(min(distances[j], distances[j + 1], new_distances[-1]) + 1)
        distances = new_distances

    return distances[-1]


def shift(x): return (x[0], x[1:])


def slurp_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def get_row(line: str) -> list[str]:
    return list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), line.split(','))))


def parse_time(s: str) -> float:
    hours = 0.0
    if ':' in s:
        time = datetime.strptime(s, "%H:%M")
        hours = time.hour + time.minute / 60.0
    else:
        hours = float(s)
    return hours


def parse_date_or_none(s: str) -> date:
    d = None
    try:
        d = datetime.strptime(s, "%d/%m/%Y").date()
    except:
        return None
    return d


def parse(content: str, real_timings: bool) -> dict:
    data = json.loads(content)

    studied_per_day = {}
    today = date.today()
    studied_today = 0.0
    total = 0.0

    subjects = dict(map(lambda x: (
        x[0], {'credits': x[1]['weight'], 'due': parse_date_or_none(x[1]['due']), 'hours': 0, 'score': 0}), data['subjects'].items()))

    first_date = date.fromtimestamp(100_000_000_000)
    for session in data['sessions']:
        subject = session['subject']
        date_ = datetime.strptime(session['date'], "%d/%m/%Y").date()

        first_date = min(date_, first_date)

        hours = session['duration']
        orig_hours = hours
        confidence = CONFIDENCE_LEVELS[session['confidence']]
        if not real_timings:
            hours *= confidence
        total += hours
        # if confidence == 0.0:
        #     hours = 0.0

        date_key = date_.strftime("%d/%m/%Y")
        if not (date_key in studied_per_day):
            studied_per_day[date_key] = 0.0
        studied_per_day[date_key] += hours

        if (today - date_).days == 0:
            studied_today += hours

        subjects[subject]['hours'] += hours
        subjects[subject]['score'] += orig_hours * confidence

    max_available_time = max([1] + list(map(lambda x: (x-first_date).days + 1, filter(lambda x: x != None, map(lambda x: x['due'], subjects.values())))))
    all_dates = list(map(lambda x: x['due'], subjects.values()))
    for s in subjects:
        subjects[s]['score'] /= subjects[s]['credits']
        if subjects[s]['due'] != None:
            if (subjects[s]['due'] - date.today()).days < 0:
                subjects[s]['score'] *= -1
            else:
                first_study_day = min(map(lambda x: datetime.strptime(x['date'], "%d/%m/%Y").date(), filter(lambda x: x['subject'] == s, data['sessions'])))
                occupied_before = len(list(filter(lambda x: x <= subjects[s]['due'] and x >= today, all_dates)))
                days_to_exam = (subjects[s]['due'] - today).days + 1
                free_days_to_exam = max(1, days_to_exam - occupied_before)
                # print(s, free_days_to_exam)
                subjects[s]['score'] *= log(free_days_to_exam)
        else:
            subjects[s]['score'] /= max_available_time

    studied_per_day = list(map(lambda x: (datetime.strptime(
        x[0], "%d/%m/%Y").date(), x[1]), studied_per_day.items()))

    dates = list(map(lambda x: x[0], studied_per_day))
    min_date = min(dates + [today])

    for i in range((today-min_date).days+1):
        d = min_date + timedelta(days=i)
        if not d in dates:
            studied_per_day.append((d, 0.0))

    studied_per_day = sorted(studied_per_day, key=lambda x: x[0])

    return {'subjects': subjects, 'total': total, 'today': studied_today, 'daily': studied_per_day}


def get_next_subject(subjects: dict) -> str:
    lowest = min(filter(lambda x: x[1]['score'] > 0, subjects.items()), key=lambda x: x[1]['score'])
    return lowest[0]


def calc_scores(subjects: dict) -> list:
    scores = list(map(lambda x: (x[0], x[1]['score']), subjects.items()))
    max_score = max(map(lambda x: x[1], scores))
    if max_score == 0:
        max_score = 1
    # avg_score = sum(map(lambda x: x[1], scores)) / len(items)
    scores = list(map(lambda x: (x[0], x[1]/max_score), scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores
    # print("\n".join(map(lambda x: f" - {x[0].ljust(max_len)}: {x[1]/max_score:.2f}", scores)))


def italic(s: str) -> str:
    return f"\033[3m{s}\033[0m"


def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def gray(s: str) -> str:
    return f"\033[90m{s}\033[0m"

def format_time(t: float, show_secs: bool = False) -> str:
    mins = (t-floor(t)) * 60
    if show_secs:
        mins = floor(mins)
    else:
        mins = round(mins)
    h = floor(t)
    if mins == 60:
        h += 1
        mins = 0
    secs = floor(((t - floor(t))*60 - mins)*60.0)
    return f"{h}:{mins:02}" + (f":{secs:02}" if show_secs else '')


def print_next_subject(file_path: str):
    content = slurp_file(file_path)
    parsed = parse(content, False)

    # print("---")
    print(f"Studied today: {format_time(
        parsed['today'])} ({parsed['today']:.2f}h)")
    next_subject = get_next_subject(parsed['subjects'])
    print(f"Your next subject should be: '{italic(next_subject)}'")


def add_study_time(file_path):
    content = slurp_file(file_path)
    data = json.loads(content)
    parsed = parse(content, False)

    subjects = sorted(data['subjects'])
    for i, s in enumerate(subjects):
        print(f"  {i+1}) {s}")

    idx = 0
    try:
        idx = int(input("Choose a subject > "))
    except:
        print("ERROR: Invalid index.")
        return

    if idx < 1 or idx > len(subjects):
        print("ERROR: Index out of bounds.")
        return

    amount = 0.0
    try:
        amount = parse_time(input("Amount of time studied [h]/[h:mm] > "))
    except:
        print("ERROR: Invalid time amount.")
        return

    confidence = max(CONFIDENCE_LEVELS.items(), key=lambda x: x[1])[0]
    for i, cl in enumerate(CONFIDENCE_LEVELS):
        if i != 0:
            print(f"  {i}) {cl}")

    confidence_idx = 1
    try:
        confidence_idx = int(input("Confidence feeling > "))
    except:
        confidence_idx = len(CONFIDENCE_LEVELS)-1
        print(f"ERROR: Invalid index, going with {confidence_idx}.")

    if idx < 0:
        print("ERROR: Index out of bounds, going with 1")
        confidence_idx = 1
    elif idx > len(subjects):
        confidence_idx = len(CONFIDENCE_LEVELS)-1
        print(f"ERROR: Index out of bounds, going with {confidence_idx}")

    date_ = date.strftime(date.today(), "%d/%m/%Y")

    data['sessions'].append({
        'subject': subjects[idx-1],
        'date': date_,
        'duration': amount,
        'confidence': CLEVELS_LIST[confidence_idx][0],
    })

    with open(file_path, 'w') as f:
        f.write(json.dumps(data))

    print()
    print("Successfully added study session!")
    print(f"Studied {format_time(
        parsed['today'] + amount*CLEVELS_LIST[confidence_idx][1])} today!")


def float_len(x): return floor(log10(max(x, 1)) + 1) + 3


def print_stats(file_path: str, real: bool = False):
    print()
    content = slurp_file(file_path)
    parsed = parse(content, real)
    scores = sorted(calc_scores(parsed['subjects']), key=lambda x: x[1])

    max_len = max(map(lambda x: len(x[0]), scores))
    max_hours = max(map(lambda x: x[1]['hours'], parsed['subjects'].items()))
    max_hours_digits = float_len(max_hours)

    min_score = min(dict(scores).values())
    max_score = max(dict(scores).values())
    padding = (1.0 - min_score) / 10.0
    (interval_min, interval_max) = (0, 1)  # (min_score-padding, 1.0)
    display_power = 2 if (max_score - min_score) < 0.75 else 1

    print("  \033[1m=== Subjects ===\033[0m")
    positive_scores = filter(lambda x: x[1] >= 0.0, scores)
    negative_scores = list(filter(lambda x: x[1] < 0.0, scores))
    for (sub, sc) in positive_scores:
        h = parsed['subjects'][sub]['hours']
        hours = f"{h:.2f}h".ljust(max_hours_digits+1)
        sub = (sub + ": ").ljust(max_len+5, '.')
        sc_str = f"{sc:.2f}".ljust(4, '0')
        n_chars = round(
            ((sc - interval_min) / (interval_max-interval_min))**display_power * BAR_LENGTH)
        # to_print = f"\033[7m\033[1m{to_print[:n_chars]}\033[0m{to_print[n_chars:]}\033[0m"
        progress_bar = "[" + "="*n_chars \
            + " "*(BAR_LENGTH-n_chars) + "]"
        to_print = f"  {sub} {hours}    {progress_bar} {sc_str}"
        print(to_print)
    if len(negative_scores) > 0:
        print(gray("  ---"))
        for (sub, sc) in negative_scores:
            sc = -sc
            h = parsed['subjects'][sub]['hours']
            hours = f"{h:.2f}h".ljust(max_hours_digits+1)
            sub = (sub + ": ").ljust(max_len+5, '.')
            sc_str = f"{sc:.2f}".ljust(4, '0')
            n_chars = round(
                ((sc - interval_min) / (interval_max-interval_min))**display_power * BAR_LENGTH)
            # to_print = f"\033[7m\033[1m{to_print[:n_chars]}\033[0m{to_print[n_chars:]}\033[0m"
            progress_bar = "[" + "="*n_chars \
                + " "*(BAR_LENGTH-n_chars) + "]"
            to_print = gray(f"  {sub} {hours}    {progress_bar} {sc_str}")
            print(to_print)

    print()
    print("  \033[1m=== Time ===\033[0m")

    # print(parsed['daily'])
    last_days_average = 0.0
    n_days = 1.0
    if len(parsed['daily']) > 0:
        today = date.today()
        studied_today = dict(map(lambda x: (x[0].strftime("%d/%m/%Y"), x[1]), parsed['daily']))[
            date.today().strftime("%d/%m/%Y")]
        DAYS = 7
        last_days = list(
            filter(lambda x: (today - x[0]).days < DAYS, parsed['daily']))
        n_days = (today - last_days[0][0]).days + 1
        last_days_hours = sum(map(lambda x: x[1], last_days))
        last_days_average = last_days_hours / n_days

    stats = [
        ("Studied in total", parsed['total']),
        (f"Last {str(n_days) + ' ' if n_days > 1 else ''}day{'s' if n_days >
         1 else ''} average", last_days_average),
        ("Today", studied_today),
    ]
    max_title_len = max(map(lambda x: len(x[0]), stats))
    max_hours_digits = float_len(max(map(lambda x: x[1], stats)))

    for (title, amount) in stats:
        stitle = (title+":").ljust(max_title_len + 1)
        samount = f"{amount:.2f}".rjust(max_hours_digits)
        hamount = format_time(amount).ljust(max_hours_digits)
        print(f"  {stitle} {hamount} ({samount})")
    print()


recording_paused = False
recording = True


def update_prompt():
    recorded_studied_time = 0.0
    was_recording_paused = recording_paused
    last_t = time()
    while recording:
        curr_t = time()
        if not recording_paused:
            recorded_studied_time += (curr_t-last_t)
        last_t = curr_t
        # Clear the current line and print the new prompt
        back = '\r'
        if was_recording_paused != recording_paused:
            was_recording_paused = recording_paused
            back = '\033[F'
        sys.stdout.write(f"{back}\033[K{format_time(recorded_studied_time/3600.0, show_secs=True)
                                        } [press enter to {'resume' if recording_paused else 'pause'}]")
        sys.stdout.flush()
        sleep(0.1)


def record_time(file_path: str):
    content = slurp_file(file_path)
    data = json.loads(content)
    parsed = parse(content, False)

    subjects = sorted(data['subjects'])
    suggested = get_next_subject(parsed['subjects'])
    for i, s in enumerate(subjects):
        to_print = ""
        if s == suggested:
            to_print = bold(f"  {i+1}) {italic(s)}")
        else:
            to_print = f"  {i+1}) {s}"
        print(to_print)

    idx = 0
    while True:
        try:
            idx = int(input(f"Choose a subject > "))
        except KeyboardInterrupt:
            print()
            return
        except:
            print("ERROR: Invalid index.")
            continue
        if idx < 1 or idx > len(subjects):
            print("ERROR: Index out of bounds.")
            continue
        break

    start = time()
    studied_time = 0.0
    global recording_paused
    # pause_start = 0
    # pause_time = 0.0
    thread = threading.Thread(target=update_prompt)
    thread.start()
    while True:
        try:
            input()
            if recording_paused:
                # pause_time += time() - pause_start
                start = time()
                recording_paused = not recording_paused
            else:
                studied_time += time() - start
                # pause_start = time()
                recording_paused = not recording_paused
        except KeyboardInterrupt:
            print()
            if recording_paused:
                # pause_time += time() - pause_start
                pass
            else:
                studied_time += time() - start
            # studied_time -= pause_time
            global recording
            recording = False
            thread.join()
            print(f"Studied time: {format_time(studied_time/3600)}h")
            break
        except:
            print("ERROR: Unknown command")
            continue

    adjust = 0
    try:
        adjust = float(input("Adjust time (amount in min) > "))/60.0
    except KeyboardInterrupt:
        print()
        return
    except:
        print("ERROR: Invalid adjust amount. Going with 0.")

    studied_time /= 3600.0
    amount = studied_time + adjust

    confidence = max(CONFIDENCE_LEVELS.items(), key=lambda x: x[1])[0]
    for i, cl in enumerate(CONFIDENCE_LEVELS):
        if i != 0:
            print(f"  {i}) {cl}")

    confidence_idx = 1
    try:
        confidence_idx = int(input("Confidence feeling > "))
    except KeyboardInterrupt:
        print()
        return
    except:
        confidence_idx = len(CONFIDENCE_LEVELS)-1
        print(f"ERROR: Invalid index, going with {confidence_idx}.")

    if idx < 0:
        print("ERROR: Index out of bounds, going with 1")
        confidence_idx = 1
    elif idx > len(subjects):
        confidence_idx = len(CONFIDENCE_LEVELS)-1
        print(f"ERROR: Index out of bounds, going with {confidence_idx}")

    date_ = date.strftime(date.today(), "%d/%m/%Y")

    data['sessions'].append({
        'subject': subjects[idx-1],
        'date': date_,
        'duration': amount,
        'confidence': CLEVELS_LIST[confidence_idx][0],
    })

    with open(file_path, 'w') as f:
        f.write(json.dumps(data))

    print()
    print("Successfully added study session!")
    print(f"Studied {format_time(
        parsed['today'] + amount*CLEVELS_LIST[confidence_idx][1])} today!")


def print_history(file_path: str, n: int, real: bool = False):
    content = slurp_file(file_path)
    # data = json.loads(content)
    parsed = parse(content, real)
    
    print()
    n = min(max(n, 0), len(parsed['daily']))
    period_str = ('full: ' if n >= len(parsed['daily']) else '') + f'last {n} day' + ('s' if n != 1 else '')
    print(f"  \033[1m=== History ({italic(period_str)}) ===\033[0m")
    
    if n == 0:
        print()
        return

    start_idx = max(len(parsed['daily']) - n, 0)
    days = parsed['daily'][start_idx:]
    max_t = max(map(lambda x: x[1], days))
    days_parsed = list(map(lambda x: (x[0].strftime("%A"), x[0].strftime(
        "%d/%m"), format_time(x[1]), f"{x[1]:.2f}", x[1]), days))
    max_wd_len = max(map(lambda x: len(x[0]), days_parsed))
    max_d_len = max(map(lambda x: len(x[1]), days_parsed))
    max_dt_len = max(map(lambda x: len(x[3]), days_parsed))
    for wd, d, ft, dt, t in days_parsed:
        n_chars = round(
            (t / max_t) * BAR_LENGTH)
        # to_print = f"\033[7m\033[1m{to_print[:n_chars]}\033[0m{to_print[n_chars:]}\033[0m"
        progress_bar = "[\033[0m" + "="*n_chars + \
            "\033[0m" + " "*(BAR_LENGTH-n_chars) + "]"
        print(f"  {wd.ljust(max_wd_len)} {d.ljust(max_d_len)}: {
              progress_bar}  {dt.rjust(max_dt_len)}h ({ft})")
    print()
    per_day_average = mean(map(lambda x: x[1], days))
    print(f"  Average over period: {format_time(per_day_average)} h/day")
    print()





def main():
    program, argv = shift(sys.argv)

    if len(argv) == 0:
        print("ERROR: CSV file not provided")
        print_help(program)
        return
    filepath, argv = shift(argv)

    real = '--real' in argv

    option = "next"
    if len(argv) > 0:
        option, argv = shift(argv)
        
    print(bold(f"===== {datetime.now().strftime('%d/%m/%y - %H:%M')} ====="))
    
    subcommands = ["add", "help", "history", "next", "record", "stats"]

    match option:
        case "add":
            add_study_time(filepath)
        case "history":
            n = 7
            if len(argv) > 0:
                sn, argv = shift(argv)
                n = int(sn)
            print_history(filepath, n, real=real)
        case "next":
            print_next_subject(filepath)
        case "record":
            record_time(filepath)
        case "stats":
            print_stats(filepath, real=real)
        case "help":
            print_help(program)
        case sc:
            xs = list(sorted(filter(lambda x: x[1] < min(map(len, subcommands)), map(lambda x: (x, lev(sc, x)), subcommands)), key=lambda x: x[1]))
            
            print(f"ERROR: Unknown subcommand `{sc}`."
                  + (f" Maybe you meant `{italic(xs[0][0])}`?" if len(xs) > 0 else "")
            )
            print("Use subcommand `help` to get the full list of subcommands.")
            return


if __name__ == '__main__':
    main()
