###############################################################################
#                                                                             #
# This is the part of the program where standard print messages are defined   #
#                                                                             #
###############################################################################

import sys
import time


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    string = "{:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes),
                                                seconds)
    return string


def msg_timestamp(s, starttime=None):
    s = s + time.strftime("%Y/%m/%d %X")
    if starttime is not None:
        s = s + "\nTime elapsed {}".format(timer(starttime, time.time()))
    return s


def msg_program_header(module: str, version: float):
    # ASCII FONTS from: http://patorjk.com/software/taag/
    # Font = "Big"
    print("###################################################")
    print(" Wellington Fast Assessment of Reactions: WellFaRe")
    print(" Module: {}".format(module))
    print("  __          __  _ _ ______      _____           ")
    print("  \ \        / / | | |  ____/\   |  __ \          ")
    print("   \ \  /\  / /__| | | |__ /  \  | |__) |__       ")
    print("    \ \/  \/ / _ \ | |  __/ /\ \ |  _  // _ \     ")
    print("     \  /\  /  __/ | | | / ____ \| | \ \  __/     ")
    print("      \/  \/ \___|_|_|_|/_/    \_\_|  \_\___|     ")
    print("                                  Version {: .2f} ".format(version))
    print("    WellFARe Copyright (C) 2018 Matthias Lein     ")
    print(" This program comes with ABSOLUTELY NO WARRANTY   ")
    print(" This is free software, and you are welcome to    ")
    print("   redistribute it under certain conditions.      ")
    msg_timestamp('Program started at: ')
    print("###################################################\n")


def msg_program_footer(starttime=None, exit_code=None):
    print("\n###################################################")
    print(msg_timestamp('Program terminated at: '))
    if starttime is not None:
        print("Total time elapsed {}".format(timer(starttime, time.time())))
    print("###################################################")
    if exit_code is not None:
        sys.exit(exit_code)


def msg_program_warning(s):
    print("\n###################################################")
    print("    __          __              _                  ")
    print("    \ \        / /             (_)                 ")
    print("     \ \  /\  / /_ _ _ __ _ __  _ _ __   __ _      ")
    print("      \ \/  \/ / _` | '__| '_ \| | '_ \ / _` |     ")
    print("       \  /\  / (_| | |  | | | | | | | | (_| |     ")
    print("        \/  \/ \__,_|_|  |_| |_|_|_| |_|\__, |     ")
    print("                                         __/ |     ")
    print("                                        |___/      ")
    msg_timestamp('Warning time/date: ')
    print(s)
    print("###################################################\n")
    sys.stdout.flush()
    return


def msg_program_error(s: str = "An error has occurred, exiting now",
                      starttime=None, destination=sys.stdout):
    print(
        "\n###################################################",
        file=destination)
    print("  ______                     ", file=destination)
    print(" |  ____|                    ", file=destination)
    print(" | |__   _ __ _ __ ___  _ __ ", file=destination)
    print(" |  __| | '__| '__/ _ \| '__|", file=destination)
    print(" | |____| |  | | | (_) | |   ", file=destination)
    print(" |______|_|  |_|  \___/|_|   ", file=destination)
    print(msg_timestamp('Error time/date: ', starttime), file=destination)
    print(s, file=destination)
    print(
        "###################################################\n",
        file=destination)
    sys.exit(-1)


def main():
    # Record start time
    prg_start_time = time.time()
    # Print some messages
    msg_program_header("Messages", 1.0)
    # msg_program_warning("Demonstration of a warning message")
    msg_program_error("Demonstration of an error message", prg_start_time)
    # msg_program_error("Error messages can also be sent to stderr",
    #                   prg_start_time,
    #                   destination=sys.stderr)
    msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
