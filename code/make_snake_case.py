import re

# This function isn't necessary, but fixes the weird casing in raw data files
def make_snake_case(original):
    """
    Convert original string to snake case.
    Because this will be used on column names of csv files, it should make minimal assumptions about input casing
    and it may grow into a monstrosity as we discover and add more edge cases :(

    Examples:
        make_snake_case("CamelCase") -> "camel_case"
        make_snake_case("Two Words") -> "two_words"
    """
    output = ""
    for letter in original:
        if letter.isupper():
            output += "_"+letter.lower()
        elif letter.isspace():  #some inputted data has spaces
            output += "_"
        else:
            output += letter
    # Remove repeated underscores introduced by strings like "Caps And Spaces"
    output_without_repeated_underscores = re.sub("_+", "_", output)
    # Remove leading and trailing underscores
    output_with_appropriate_underscoring = re.sub("^_|_$", "", output_without_repeated_underscores)
    return(output_with_appropriate_underscoring)
