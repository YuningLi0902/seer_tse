import csv
import json
import openpyxl
import os
import pickle
import xlsxwriter


def GetFileBySuffix(patch_dir, patch_suffix=".patch"):
    patch_files = []
    for root, dirs, files in os.walk(patch_dir):
        for file in files:
            if file.endswith(patch_suffix):
                patch_files.append(os.path.join(root, file))
    return patch_files


def ReadFileContent(file_name, content="raw", method="r"):
    if method not in ["r", "rb"]:
        method = "r"
    with open(file_name, method) as f:
        if content == "raw":
            return f.read()
        else:
            return f.readlines()


def LoadJsonFile(file_name, method="r"):
    if method not in ["r", "rb"]:
        method = "r"
    with open(file_name, method) as f:
        json_content = json.load(f)
    return json_content


def MakeCsv(zip_list, target_file):
    with open(target_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip_list)
    return 0


def ReadXLSX(source_file, escape_row=0, escape_column=0, method="column"):
    workbook = openpyxl.load_workbook(source_file)
    sheet = workbook.active
    result = []
    if method == "column":
        for i, column in enumerate(sheet.columns):
            if i < escape_column:
                continue
            result.append([item.value for item in list(column)[escape_row:]])
    else:
        for i, row in enumerate(sheet.rows):
            if i < escape_row:
                continue
            result.append([item.value for item in list(row)[escape_column:]])
    return result, workbook


def CreateActiveSheet(engine="openpyxl", save_output=None):
    if engine == "xlsxwriter":
        if save_output:
            return xlsxwriter.Workbook(save_output)
        else:
            print("WARNING: Using xlsxwriter as engine, but missing output file name.")
            return None
    else:
        return openpyxl.Workbook()


def XlsxWriterSave(output_dir, output_content, start_row=0, start_col=0):
    workbook = CreateActiveSheet(engine="xlsxwriter", save_output=output_dir)
    sheet = workbook.add_worksheet()
    for i, row in enumerate(output_content):
        sheet.write_row(start_row + i, start_col, row)
    workbook.close()


def DumpJson(data, target, binary=False, encoding=None, string_format=False, all_ascii=False, indent=0):
    if encoding:
        with open(target, "wb" if binary else "w", encoding=encoding) as f:
            if string_format:
                json.dump(data, f, ensure_ascii=all_ascii, indent=indent)
            else:
                json_string = json.dumps(data, ensure_ascii=all_ascii, indent=indent)
                f.write(json_string)
    else:
        with open(target, "wb" if binary else "w") as f:
            if string_format:
                json.dump(data, f, ensure_ascii=all_ascii, indent=indent)
            else:
                json_string = json.dumps(data, ensure_ascii=all_ascii, indent=indent)
                f.write(json_string)


def LoadJson(source, binary=False, encoding=None):
    if not os.path.isfile(source):
        print("WARNING: Invalid source file.")
        return None
    if encoding:
        with open(source, "rb" if binary else "r", encoding=encoding) as f:
            return json.load(f)
    else:
        with open(source, "rb" if binary else "r") as f:
            return json.load(f)


def Xlsx2Json(source, target, have_title=True, keys=None, readable=False):
    if not source.endswith(".xlsx") or not os.path.exists(source):
        print("WARNING: Invalid source file.")
        return None
    xlsx_content, _ = ReadXLSX(source, method="row")
    if have_title:
        keys = xlsx_content[0]
        xlsx_content = xlsx_content[1:]
    if keys is None:
        print("WARNING: Detect no keys, saving in the list format.")
        output_content = xlsx_content
    else:
        if len(keys) != len(xlsx_content[0]):
            print("WARNING: The numbers of keys and values do not match.")
            return None
        output_content = []
        for item in xlsx_content:
            json_element = dict()
            for i in range(len(keys)):
                json_item = {keys[i]: item[i]}
                json_element.update(json_item)
            output_content.append(json_element)
    if readable:
        DumpJson(output_content, target, string_format=True, indent=2)
    else:
        DumpJson(output_content, target)


def LoadPickleFile(file_dir):
    with open(file_dir, "rb") as f:
        return pickle.load(f)


def DumpPickleFile(target_dir, content):
    with open(target_dir, "wb") as f:
        pickle.dump(content, f)


def WriteFile(file_name, content, method="w"):
    with open(file_name, method) as f:
        f.write(content)
