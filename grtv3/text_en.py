"""受試者看到的文字（英文；兩支原腳本的文字本來就都是英文）。"""
INTRO = ("Welcome, and thank you for taking part.\n\nThis session has two parts: a short calibration, then the main "
         "experiment. Please make sure you are wearing the headphones before continuing.\n\nPress the space bar to continue.")

INSTRUCTION = ("The task.\n\nFour coloured squares will appear one at a time, one in each corner of the screen, and each of "
               "them comes with its own speech sound. Try to remember which colour and which sound belong together and where "
               "they appeared.\n\nA frame will then appear at one of the corners, and an item will appear in it. When you see "
               "the frame, recall what has just appeared. On the final screen four options are shown, one in each corner - "
               "Each option is a colour with a syllable label, [bi] or [pi], underneath - choose the one you have in mind by "
               "pressing the key for its position:\n\n        i = upper left            o = upper right\n        k = lower left"
               "            l = lower right\n\nThe picture below shows how the four keys point to the four corners.\n\nThe "
               "session has four parts: colour calibration, sound calibration, a short practice, then the main experiment in "
               "four blocks with a rest after each one. During calibration and practice the correct option is framed after each "
               "answer. The colours and sounds are adjusted to stay difficult, so feeling unsure is normal - just give your best "
               "guess every time. Answer as accurately as you can - speed is not important.\n\nPress the space bar to begin.")

INSTRUCTION_JOINT = INSTRUCTION.replace(
    "The session has four parts: colour calibration, sound calibration, a short practice,",
    "The session has three parts: a calibration in which both the colours and the sounds are adjusted, a short practice,")

KEY_MAP_LINES = ("        i = upper left            o = upper right\n"
                 "        k = lower left            l = lower right\n\n")

FEEDBACK = {"correct": "Correct", "noresp": "No response", "wrong": "Wrong"}
FEEDBACK_SUFFIX = "\nThe correct option is framed."


def rest_after_colour():
    return ("Part 1 (colour calibration) finished.\n\nPart 2 is the same task; this time the sounds are adjusted and all "
            "the squares have the same colour.\n\nTake a short rest.\n\nPress the space bar when you are ready to continue.")


def rest_after_joint(n_practice):
    return ("Calibration finished.\n\n"
            f"Next is a short practice of {n_practice} trials with the colours and sounds set for you. The correct option is "
            "still framed after each answer.\n\nPress the space bar when you are ready to continue.")


def rest_after_sound(n_practice):
    return ("Part 2 (sound calibration) finished.\n\n"
            f"Next is a short practice of {n_practice} trials with the colours and sounds set for you. The correct option is "
            "still framed after each answer.\n\nPress the space bar when you are ready to continue.")


def rest_after_practice(practice_correct, n_practice, n_blocks):
    pct = 100.0 * practice_correct / max(n_practice, 1)
    return ("Practice finished.\n\n"
            f"You got {practice_correct} of {n_practice} correct ({pct:.0f}%).\n\n"
            "If that felt like guessing, tell the experimenter now.\n\n"
            "The main experiment starts next. There is no more feedback. "
            f"It is divided into {n_blocks} blocks with a rest after each one.\n\n" + KEY_MAP_LINES + "Press the space bar to begin.")


def rest_between_blocks(done_blocks, n_blocks):
    return (f"Block {done_blocks} of {n_blocks} finished.\n\nTake a rest. Look away from the screen for a moment.\n\n"
            + KEY_MAP_LINES + "Press the space bar when you are ready to continue.")
