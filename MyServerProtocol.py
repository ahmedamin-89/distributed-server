def BuildMessage(r):
    return str(r)


def ParseMessage(st):
    st_list = st.split(',')
    if len(st_list) < 3:
        return "badop", 0, 0
    return str(st_list[0]), int(st_list[1]), int(st_list[2])
