"""
Verbose printing
- - - - - 
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong (@Leedehai)
Date: March 1, 2017
- - - - -
This function handles printing verbose printing. The easiest part :)
"""

def vprint(if_print, s=None, color=None):
    '''
    This function is called by external code.
    Params:
      if_print: a boolean. If True, then print.
      color: can be None, or color names. If a 
        color name is not supported then print in normal color.
      s: string to be printed.
    Returns:
      (empty)
    '''
    if isinstance(if_print, bool) == False:
        # The user did not give the if_print flag or gave a wrong type.
        raise ValueError("The if_print flag should be a bool, but it is: " + str(if_print))
    if if_print == True:
        _process(s, color=color)
        return
    else:
        return

def _process(s, color=None):
    '''
    Some processing. Internal function, do not use.
    '''
    try:
        s = str(s)
    except:
        raise ValueError("The color option provided is not string")

    _verbose_print(s, color=color)
    return
    
def _verbose_print(s, color):
    '''
    Printing. Internal function, do not use.
    colors:
      GRAY, GREY = GRAY, RED (r), GREEN (g), YELLOW, BLUE (b), MAGENTA (m), MAG = MAGENTA, CYAN (c), WHITE (w)
    '''
    _START = "\033["

    GRAY = _START + "30m"
    RED = _START + "31m"
    GREEN = _START + "32m"
    YELLOW = _START + "33m"
    BLUE = _START + "34m"
    MAGENTA = _START + "35m"
    CYAN = _START + "36m"
    WHITE = _START + "37m"
    CRIMSON = _START + "38m"

    END = "\033[m"

    if color != None:
        color = color.upper()

    if color == 'GRAY' or color == 'GREY':
        print GRAY + s + END
        return
    if color == 'RED' or color == 'R':
        print RED + s + END
        return
    elif color == 'GREEN' or color == 'G':
        print GREEN + s + END
        return
    elif color == 'YELLOW':
        print YELLOW + s + END
        return
    elif color == 'BLUE' or color == 'B':
        print BLUE + s + END
        return
    elif color == 'MAGENTA' or color == 'MAG' or color == 'M':
        print MAGENTA + s + END
        return
    elif color == 'CYAN' or color == 'C':
        print CYAN + s + END
        return
    elif color == 'WHITE' or color == 'W':
        print WHITE + s + END
        return
    elif color == 'CRIMSON':
        print CRIMSON + s + END
        return
    else: # color is None or is not supported
        print s
        return

if __name__ == "__main__":
    print "If if_print == False:"

    print "Should print nothing: ",
    vprint(False, "hello", 'GREEN')
    print ""

    print "If if_print == True:"

    print "Should be gray: ",
    vprint(True, "Over hill, over dale,", 'GRAY')
    print "Should be gray: ",
    vprint(True, "Thorough bush, thorough brier,", 'GREY')
    print "Should be red: ",
    vprint(True, "Over park, over pale,", 'r')
    print "Should be green: ",
    vprint(True, "Thorough flood, thorough fire!", 'GReeN')
    print "Should be yellow: ",
    vprint(True, "I do wander everywhere,", 'YELLOW')
    print "Should be blue: ",
    vprint(True, "Swifter than the moon's sphere;", 'BLUE')
    print "Should be magenta: ",
    vprint(True, "And I serve the Fairy Queen,", 'MAGENTA')
    print "Should be magenta: ",
    vprint(True, "To dew her orbs upon the green;", 'MAg')
    print "Should be cyan: ",
    vprint(True, "The cowslips tall her pensioners be;", 'C')
    print "Should be white: ",
    vprint(True, "In their gold coats spots you see;", 'WHITE')
    print "Should be normal: ",
    vprint(True, "Those be rubies, fairy favours;", 'unkown_color')
    print "Should be normal: ",
    vprint(True, "In those freckles live their savours;", None)
    print "Should be normal: ",
    vprint(True, "I must go seek some dewdrops here,")
    print "Should be blue: ",
    str1 = "And hang a pearl"
    str2 = " in every cowslip's ear."
    vprint(True, str1 + str2, 'b')
    print "Should be a number in green: ",
    vprint(True, 5*8+2, 'g')
    print "Should be a number in normal: ",
    vprint(True, 4.0)
    print __file__ + " test done. You should manually check the colors."
    # Test passed. 20:48 Mar 6, 2017