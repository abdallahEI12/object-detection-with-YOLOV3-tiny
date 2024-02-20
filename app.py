from detector import Detector

def main():
    folder_path = r'D:\educational\projects\tota\pic'
    d = Detector()
    test_data = d.get_blobs_and_classes(folder_path)
    d.detect(test_data)
    d.evaluate()


if '__main__' in __name__:
    main()