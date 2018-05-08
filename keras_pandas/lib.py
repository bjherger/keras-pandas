def check_variable_list_are_valid(variable_type_dict):
    """

    :param variable_type_dict:
    :return:
    """
    for outer_key, outer_value in variable_type_dict.items():
        for inner_key, inner_value in variable_type_dict.items():

            # Do not compare variable types to themselves
            if inner_key == outer_key:
                continue

            else:
                intersection = set(outer_value).intersection(set(inner_value))
                if len(intersection) > 0:
                    raise ValueError('Variable lists {} and {} overlap, and share key(s): {}'.
                                     format(inner_key, outer_key, intersection))

    return True

