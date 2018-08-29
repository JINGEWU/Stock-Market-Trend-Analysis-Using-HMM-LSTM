def form_index(lengths, index):
    # input:
    #     lengths
    #     index: lengths的第几个
    # output:
    #     begin_index
    #     end_index

    begin_index = sum(lengths[0:index])
    end_index = begin_index + lengths[index]

    return begin_index, end_index
