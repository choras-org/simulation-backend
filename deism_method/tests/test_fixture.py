import os


def test_create_tmp_file(create_temporary_input_file):
    """Test the creation of a temporary input file."""
    directory = os.path.dirname(create_temporary_input_file)

    assert os.path.exists(create_temporary_input_file)
    assert os.path.exists(
        os.path.join(directory, "test_room_Deism.geo"))
    assert os.path.exists(
        os.path.join(directory, "test_room_Deism.msh"))
