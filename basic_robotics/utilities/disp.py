import time
import datetime

TM_SYMBOLS = ['Xm', 'Ym', 'Zm', 'Xr', 'Yr', 'Zr']
WR_SYMBOLS = ['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz']

def disp(matrix, title = "MATRIX", nd = 3, mode = 0, pdims = True, noprint = False):
    """
    Drop in replacement for python print. Operates like Matlab's disp() function.
    Takes in an object to print, a title, and an optional mode
    Args:
        matrix: the item to be printed (does not have to be a matrix)
        title: An optional title or caption to be applied to the printed item
        nd: number of decimal places
        mode: whether or not to print the context in latex table format
        pdims: print dimensions
        noprint: simply return a string without printing anything
    Returns:
        String
    """
    matstr = ""
    if mode == 0:
        matstr = dispa(matrix, title, nd, pdims)[:-1]
    else:
        matstr = disptex(matrix, title, nd)[:-1]
    if not noprint:
        print(matstr)
    return matstr

def dispa(matrix, title = "MATRIX", nd = 3, pdims = True, h="", new = True):
    """
    Helper function for disp, used recursively
    Args:
        matrix: item to be printed
        nd: number of decimal places
        pdims: print dimensions
        h: existing string
        new: if this is a new call
    Returns:
        String
    """
    t_bar = ""
    t_tl = "╔"
    t_bl = "╚"
    #╚╔╝╗║═ Symbols Required

    #Accounts for Even or Odd amounts of letters in a title
    if (len(title) % 2 == 0):
        t_tl = t_tl + "═"
        t_bl = t_bl + "═"

    strr = ""
    #Accounts for a List of Objects, Calls itself Recursively
    if hasattr(matrix, 'TM'):
        return dispa(matrix.TAA, title)
    if isinstance(matrix, list):
        alltf = True
        all_wrench = True
        for mat in matrix:
            if not hasattr(mat, 'TM'):
                alltf = False
            if not hasattr(mat, 'wrench_arr'):
                all_wrench = False
        if alltf:
            return printTFlist(matrix, title, nd)
        if all_wrench:
            return printTFlist(matrix, title, nd, tm_names = False)

        i = 0
        str1 = (t_tl + "════════════" + " " + title + " BEGIN " + "════════════" + "╗\n")
        strm = ""
        for mat in matrix:
            if not isinstance(mat, list) and not isinstance(mat, tuple) and hasattr(matrix, 'TM'):
                strm += (str(mat) + "\n")
            else:
                if pdims:
                    strm+=("Dim " + str(i) + ":\n")
                strm += dispa(mat)
            i = i + 1
        str2 = (t_bl + t_bar + "════════════" + title + " END ═" + "════════════" + "╝\n")
        return str1 + strm + str2;

    shape = 0

    #Variety of try catch to prevent crashes due to liberal use of disp()
    try:
        try:
            shape = matrix.shape
        except:
            #Panda obects IIRC use shape as a method
            shape = matrix.shape()

        dims = len(shape)
        if dims >= 2:
            t_key = shape[dims - 1]
        else:
            t_key = max(shape)
            if new and title != "MATRIX":
                strr+= title + ": "
    except:
        #If all else fails, print Normally
        if title != "MATRIX":
            strr += (title + ": ")
        strr += (str(matrix) + "\n")
        return strr
    #Formats correct number of top and bottom markers for t_bar
    while(len(title) + 8 + (len(t_bar) * 2)) < (t_key * (nd + 7) ):
        t_bar = t_bar + "═"

    #Prints a single Dimension Vector
    if dims == 1:
        cn = 0
        if h == "╔ ":
            cn = 1
        elif h == "╚ ":
            cn = 2
        else:
            h = h + "║ "
        for i in range(shape[0]):
            t_nd = nd
            if (abs(matrix[i]) >= 9999):
                nm = len(str(abs(round(matrix[i]))))
                while t_nd > 0 and nm > 6:
                    t_nd = t_nd - 1
                    nm = nm - 1
            fmat = "{:" + str(nd + 6) +"." + str(t_nd) + "f}"


            h = h + fmat.format(matrix[i])
            if i != shape[0] - 1:
                h = h + ","

        if cn == 0:
            h = h + " ║"
        elif cn == 1:
            h = h + " ╗"
        else:
            h = h + " ╝"

        strr+= (str(h) + "\n")

    #Prints traditional Square Matrix, allows for title
    elif dims == 2:
        if title != "MATRIX":
            strr+=(t_tl + t_bar + " " + title + " BEGIN " + t_bar + "╗\n")
        for i in range(shape[0]):
            if i == 0:
                strr += dispa(matrix[i,], nd = nd, h = "╔ ", new = False)
            elif i == shape[0] - 1:
                strr += dispa(matrix[i,], nd = nd, h = "╚ ", new = False)
            else:
                strr += dispa(matrix[i,], nd = nd, new = False)
        if title != "MATRIX":
            strr+=(t_bl + t_bar + "═ " + title + " END ═" + t_bar + "╝\n")

    #Prints 3D Matrix by calling 2D recursively
    elif dims == 3:
        strr += (t_tl + t_bar + " " + title + " BEGIN " + t_bar + "╗\n")
        for i in range(shape[0]):
            if pdims:
                strr += ("DIM " + str(i) + ":\n")
            strr += dispa(matrix[i,], nd = nd, new = False)
        strr += (t_bl + t_bar + "═ " + title + " END ═" + t_bar + "╝\n")

    #Prints 4D Matrix by calling 3D recursively
    elif dims == 4:
        strr += (t_tl + t_bar + "══ " + title + " BEGIN ══" + t_bar + "╗\n")
        for i in range(shape[0]):
            strr += dispa(matrix[i,], nd = nd, title = title + " d:" + str(i), pdims = pdims, new = False)
        strr += (t_bl + t_bar + "═══ " + title + " END ═══" + t_bar + "╝\n")
    else:
        taux = "═" * (dims - 3)**2
        strr += (t_tl + t_bar + taux +" " + title + " BEGIN " + taux + t_bar + "╗\n")
        for i in range(shape[0]):
            strr += dispa(matrix[i,], title = title + " s" + str(i), new = False)
        strr += (t_bl + t_bar + taux + "═ " + title + " END ═" + taux + t_bar + "╝\n")
    return strr
    #More dimensions can be added as needed if neccessary

def disptex(matrix, title,  nd = 3, pdims = True, h=""):
    """
    Prints a matrix in latex format.
    Args:
        matrix: matrix to be printed
        title: caption
        nd: number of decimal places to round to
        pdims: print dimensions
        h: existing string
    Returns:
        String
    """
    try:
        shape = matrix.shape
    except:
        return dispa(matrix, title)
    strr = "\\begin{table}\n\\centering\n\\begin{tabular}{|"
    for i in range(shape[1]):
        strr = strr + " c "
        if i == shape[1] - 1:
            strr = strr + ("|}\n\\hline\n")
    strr+="\\toprule\n"
    strr+="%INSERT CAPTIONS HERE\n"
    strr+="\\midrule\n"
    for i in range(shape[0]):
        #strr+= "\\hline\n"
        for j in range(shape[1]):
            strr+= str(round(matrix[i, j], nd))
            if j != shape[1] - 1:
                strr+=" & "
                continue
            else:
                break
        strr+="\\\\\n"
    strr+="\\bottomrule\n"
    strr+="\\end{tabular}\n\\caption{" + title + "}\n\\end{table}\n"
    return strr

def printTFlist(matrix, title, nd, print_names = True, tm_names=True):
    """
    Prints a list of TM objects (TF was deprecated)
    Args:
        matrix: list of tms to be printed
        title: caption
        nd: number of decimal places to round to
    Returns:
        String
    """
    ico_list = TM_SYMBOLS
    if not tm_names:
        ico_list = WR_SYMBOLS
    print_saver = ""
    if print_names:
        print_saver = "═══"
    nTF = len(matrix)
    title_len = len(title)

    title_len_base = 2 * nTF * (nd+1) + (2*nTF+1)
    tLen = title_len_base + len(print_saver)
    until_title = round(title_len_base/2 - title_len/2 - 1)
    strr =  ("╔" + "═" * until_title + " " + title + " " 
            +  "═" * (tLen - until_title - 2 - title_len) + "╗\n")
    if print_names:
        strr+= "╠══╦═"
    else:
        strr+= "╠═"
    for i in range(nTF):
        #strr +=  "╔"
        strr += "═" * (nd + 6)
        if i != nTF - 1:
            strr += "╦"
    strr+= "═╣\n"
    for j in range(6):
        if print_names:
            strr+= "║" + ico_list[j] + "║ "
        else:
            strr+= "║ "
        for i in range(len(matrix)):
            t_nd = nd
            if (abs(matrix[i][j]) >= 9999):
                nm = len(str(abs(round(matrix[i][j]))))
                while t_nd > 0 and nm > 6:
                    t_nd = t_nd - 1
                    nm = nm - 1
            fmat = "{:" + str(nd + 6) +"." + str(t_nd) + "f}"

            strr = strr + fmat.format(matrix[i][j])
            if i != nTF - 1:
                strr = strr + ","
        strr+=" ║\n"
    if print_names:
        strr+= "╚══╩═"
    else:
        strr+= "╚═"
    for i in range(nTF):
        strr += "═" * (nd + 6)
        if i != nTF - 1:
            strr += "╩"
    
    strr+= "═╝\n"
    return strr

def progressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', ETA=None):
    """
    Prints a progress bar, can use ETA.
    Adapted from here: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Params:
        iteration: current iteration
        total: goal number of iterations
        prefix: Optional- Text to append to the beginning
        suffix: Optional - Text to append to the end (overwritten by ETA)
        decimals: Optional - Decimals to round to
        length: Optional - Length of progress bar in characters
        fill: Optional - Fill Character
        ETA: Optional - Time in seconds since start, triggers printing ETA in suffix
    """
    if ETA is not None:
        current = time.time()
        est_complete = (current-ETA)/(iteration+1)*(total-iteration)+current
        est_complete_str = datetime.datetime.fromtimestamp(est_complete).strftime('ETA: %Y-%m-%d %I:%M:%S%p')
        suffix = est_complete_str
    percent = ("{0:." + str(decimals) + "f}").format(100*(iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print("")
