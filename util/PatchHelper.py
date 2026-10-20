import os
import re

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from util.FileHelper import GetFileBySuffix, ReadFileContent


def ExtractPatchInfo(patch_path, dataset_name, detail=False):
    directory = patch_path.split('/')
    root_index = directory.index(dataset_name)
    inner_directory = directory[root_index:]

    if len(inner_directory) == 5:
        correctness = 1 if inner_directory[1] == "correct" else 0
    else:
        if "correct" in patch_path:
            correctness = 1
        elif "overfitting" in patch_path:
            correctness = 0
        else:
            correctness = -1
    patch_info = inner_directory[-1].split('.')[0].split('-')
    assert len(patch_info) >= 4
    proj = patch_info[1]
    bug_id = patch_info[2]
    tool = patch_info[3]
    if detail:
        patch_id = patch_info[0][5:]
        return correctness, proj, bug_id, tool, patch_id
    else:
        return correctness, proj, bug_id, tool


def CheckValidPatch(string_list):
    for string in string_list:
        if string.startswith("+") or string.startswith("-"):
            return True
    return False


def FormatPatchContent(content, no_diff_head=False, no_comment=False, no_blank=False):
    if type(content) == list:
        patch_lines = content
    elif type(content) == str:
        patch_lines = [item if item.endswith('\n') else item + '\n' for item in content.split('\n')]
    else:
        print("Unknown type of patch!")
        return content
    diff_label = False
    end_of_comment = True
    clean_lines = []
    split_line_number = -1
    clean_line_number = 0
    for i in range(len(patch_lines)):
        ori_line = patch_lines[i]
        line = ori_line.strip()
        if "*/" in line and no_comment:
            if not end_of_comment or "/*" in line:
                end_of_comment = True
            else:
                clean_lines = clean_lines[:split_line_number]
                clean_line_number = max(split_line_number, 0)
            continue
        if line != '':
            if (line.startswith("diff") or line.startswith("Index:")) and no_diff_head:
                diff_label = True
            elif line.startswith("@@"):
                if no_comment:
                    end_of_comment = True
                if no_diff_head:
                    diff_label = False
                    last_slice = clean_lines[split_line_number:clean_line_number]
                    if len(last_slice) > 0 and not CheckValidPatch(last_slice):
                        clean_lines = clean_lines[:split_line_number]
                        clean_line_number = max(split_line_number, 0)
                    else:
                        split_line_number = clean_line_number
                else:
                    clean_lines.append(line if no_blank else ori_line)
                    clean_line_number += 1
            elif not end_of_comment and no_comment:
                continue
            elif '/*' in line and no_comment:
                if "*/" in line:
                    end_of_comment = True
                else:
                    end_of_comment = False
                continue
            elif (diff_label or line.startswith("+++") or line.startswith("---")) and no_diff_head:
                continue
            elif line.startswith('//') and no_comment:
                continue
            elif line.startswith("+") or line.startswith("-"):
                if (line[1:].strip().startswith('//') or len(line) == 1) and no_comment:
                    continue
                clean_lines.append(line if no_blank else ori_line)
                clean_line_number += 1
            else:
                clean_lines.append(line if no_blank else ori_line)
                clean_line_number += 1
        else:
            if not no_blank:
                clean_lines.append(ori_line)
                clean_line_number += 1
    last_slice = clean_lines[split_line_number:clean_line_number]
    if len(last_slice) > 0 and not CheckValidPatch(last_slice):
        clean_lines = clean_lines[:split_line_number]
    if no_blank:
        return "\n".join(clean_lines)
    else:
        return "".join(clean_lines)


def FormatPatchContentWithUnidiff(content, no_diff_head=False, no_comment=False, no_blank=False):
    if type(content) == list:
        patch_str = '\n'.join(content)
    elif type(content) == str:
        patch_str = content
    else:
        print("Unknown type of patch!")
        return content
    from unidiff import PatchSet
    patch_content = PatchSet.from_string(patch_str)
    format_lines = []
    multi_line_comment_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)
    for patched_file in patch_content:
        diff_head_info = [f"--- {patched_file.source_file}\t{patched_file.source_timestamp}",
                          f"+++ {patched_file.target_file}\t{patched_file.target_timestamp}"]
        if patched_file.patch_info is not None:
            diff_head_info = [item.strip() for item in patched_file.patch_info] + diff_head_info
        for hunk in patched_file:
            hunk_lines = hunk.__str__().split('\n')
            diff_head_info.append(hunk_lines[0])
            if not no_diff_head:
                format_lines.extend(diff_head_info)
            diff_head_info.clear()
            hunk_lines = hunk_lines[1:]
            if no_comment:
                hunk_string = '\n'.join(hunk_lines)
                hunk_string_wo_multi_line_comment = multi_line_comment_pattern.sub('', hunk_string)
                hunk_lines = [RemoveSingleLineComment(item) for item in hunk_string_wo_multi_line_comment.split('\n')]
            clean_lines = []
            for i in range(len(hunk_lines)):
                ori_line = hunk_lines[i]
                line = ori_line.strip()
                if line != '':
                    if ("/*" in line or "*/" in line) and no_comment:
                        if "*/" in line:
                            clean_lines.clear()
                            if line.endswith("*/"):
                                continue
                            soc_ori_line = ori_line.index("*/") + 2
                            soc_line = line.index("*/") + 2
                            line = line[soc_line:]
                            ori_line = ori_line[soc_ori_line:]
                        if "/*" in line:
                            if line.startswith("/*"):
                                break
                            soc_ori_line = ori_line.index("/*")
                            soc_line = line.index("/*")
                            line = line[:soc_line]
                            ori_line = ori_line[:soc_ori_line]
                            clean_lines.append(line if no_blank else ori_line)
                            break
                        else:
                            clean_lines.append(line if no_blank else ori_line)
                    elif line.startswith("+") or line.startswith("-"):
                        if len(line) == 1 and no_blank:
                            continue
                        clean_lines.append(line if no_blank else ori_line)
                    else:
                        clean_lines.append(line if no_blank else ori_line)
                else:
                    if not no_blank:
                        clean_lines.append(ori_line)
            if len(clean_lines) > 0 and CheckValidPatch(clean_lines):
                format_lines.extend(clean_lines)
    return '\n'.join(format_lines)


def LoadPatches(dataset, prefix="datasets/", no_diff_head=False):
    results = []
    dataset_all_path = []
    if dataset in ["Lin2022TOSEM"]:
        dataset_correct = GetFileBySuffix(os.path.join(prefix, dataset, "correct"))
        dataset_overfitting = GetFileBySuffix(os.path.join(prefix, dataset, "overfitting"))
        dataset_all_path = dataset_correct + dataset_overfitting
    for path in dataset_all_path:
        patch = ReadFileContent(path)
        if no_diff_head:
            patch = FormatPatchContent(patch, no_diff_head=no_diff_head)
        results.append([path, patch])
    return results


def GetAllPatches(patch_dir, patch_suffix=".patch"):
    patch_files = []
    for root, dirs, files in os.walk(patch_dir):
        for file in files:
            if file.endswith(patch_suffix):
                patch_files.append(os.path.join(root, file))
    return patch_files


def GetModifiedPosition(patch_text, target="buggy", suffix=".java"):
    pattern = r"\b\d+\b"
    patch_lines = patch_text.split('\n')
    if target == "patch":
        start_index = 2
        length_index = 3
        file_indicator = "+++"
    else:
        start_index = 0
        length_index = 1
        file_indicator = "---"
    results = dict()
    file_path = ""
    for line in patch_lines:
        if line.startswith(file_indicator):
            file_path = line[3:].split(suffix)[0] + suffix
            while not file_path.startswith("src") and not file_path.startswith("source"):
                file_path = file_path[1:]
            assert file_path != ""
            results.update({file_path: []})
        elif line.startswith("@@"):
            match = re.findall(pattern, line)
            start = int(match[start_index])
            end = start + int(match[length_index])
            results[file_path].append([start, end])
    return results


def GetSeparateModification(patch_content):
    add_lines = []
    del_lines = []
    patch_content_lines = patch_content.split('\n')
    partial_add = []
    partial_del = []
    first_flag = True
    for line in patch_content_lines:
        if line.startswith("@@"):
            if first_flag:
                first_flag = False
            else:
                add_lines.append(partial_add)
                partial_add = []
                del_lines.append(partial_del)
                partial_del = []
        elif line.startswith("+") and not line.startswith("+++"):
            partial_add.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            partial_del.append(line[1:].strip())
    add_lines.append(partial_add)
    del_lines.append(partial_del)
    return add_lines, del_lines


def RemoveFunctionHeadComment(function_text):
    text_lines = function_text.split('\n')
    new_start = 0
    while True:
        line = text_lines[new_start].strip()
        if line == "" or line.startswith("//"):
            new_start += 1
        elif line.startswith("/*"):
            while not text_lines[new_start].strip().endswith("*/"):
                new_start += 1
            new_start += 1
        else:
            break
    clean_function = '\n'.join(text_lines[new_start:])
    return clean_function


def RemoveSingleLineComment(java_line):
    in_string = False
    string_quote = None
    escape = False
    for i, char in enumerate(java_line):
        if not in_string:
            if char == '/' and i + 1 < len(java_line):
                next_char = java_line[i + 1]
                if next_char == '/':
                    return java_line[:i].rstrip()
                elif next_char == '*':
                    return java_line[:i].rstrip()
            if char in ('"', "'"):
                in_string = True
                string_quote = char
        else:
            if char == string_quote and not escape:
                in_string = False
            escape = (char == '\\' and not escape)
    return java_line.rstrip()


def GetSubPatch(patch_content):
    diff_labels = ["@@", "diff", "Index:", "+++", "---"]
    sub_patches = []
    lines = patch_content.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith("@@"):
            sub_patch = []
            i += 1
            while i < len(lines) and (True not in [lines[i].startswith(label) for label in diff_labels]):
                sub_patch.append(lines[i])
                i += 1
            sub_patches.append('\n'.join(sub_patch))
        else:
            i += 1
    return sub_patches


def GetMinimumIndentSpaceNumber(content, is_patch=False):
    max_indent = 10000
    min_indent = 0
    indent = max_indent
    content_lines = content.split('\n')
    for line in content_lines:
        if indent == min_indent:
            break
        if line.strip() == "":
            continue
        if is_patch and (line.startswith('+') or line.startswith('-')):
            line = line[1:]
            spaces = len(line) - len(line.lstrip()) + 1
        else:
            spaces = len(line) - len(line.lstrip())
        if spaces < indent:
            indent = spaces
    if indent == max_indent:
        indent = min_indent
    return indent


def ReplaceLinePrefixTab(line_content, replacement="    ", is_patch=False):
    if line_content.strip() == "":
        return line_content
    if is_patch and (line_content.startswith('+') or line_content.startswith('-')):
        white_space = 1
    else:
        white_space = 0
    while white_space < len(line_content) and line_content[white_space].isspace():
        white_space += 1
    space_prefix = line_content[:white_space].replace('\t', replacement)
    concrete_code = line_content[white_space:]
    no_tab_line = space_prefix + concrete_code
    return no_tab_line


def RemovePatchRedundantIndent(patch_content, redundant_spaces=0):
    if redundant_spaces == 0:
        return patch_content
    content_lines = patch_content.split('\n')
    new_lines = []
    for line in content_lines:
        if line.strip() == "" or line[1:].strip() == "":
            new_lines.append(line)
            continue
        if line.startswith('+') or line.startswith('-'):
            clean_line = line[1:]
            if redundant_spaces > 0:
                for i in range(redundant_spaces):
                    if clean_line[0].isspace():
                        clean_line = clean_line[1:]
                    else:
                        break
                new_lines.append(line[0] + clean_line)
            else:
                new_lines.append(line[0] + ' ' * redundant_spaces + clean_line)
        else:
            clean_line = line
            if redundant_spaces > 0:
                for i in range(redundant_spaces):
                    if clean_line[0].isspace():
                        clean_line = clean_line[1:]
                    else:
                        break
                new_lines.append(clean_line)
            else:
                new_lines.append(' ' * redundant_spaces + clean_line)
    return '\n'.join(new_lines)


def ConfirmDiffPadding(patch_content):
    patch_content_lines = patch_content.split('\n')
    head_padding = 100000
    tail_padding = -1
    for i in range(len(patch_content_lines)):
        line = patch_content_lines[i]
        if line.startswith('+') or line.startswith('-'):
            if head_padding > i:
                head_padding = i
            if tail_padding < i:
                tail_padding = i
    tail_padding = len(patch_content_lines) - tail_padding - 1
    return head_padding, tail_padding


def ExpandFunctionPatch(buggy_function, start_no, patch_info):
    buggy_function_lines = buggy_function.split('\n')
    line_numbers = [i for i in range(start_no, start_no + len(buggy_function_lines))]
    try:
        numbered_lines = [[no, line] for no, line in zip(line_numbers, buggy_function_lines)]
        for item in patch_info:
            patch_lines = item[1].split('\n')
            patch_lines = [ReplaceLinePrefixTab(pl, is_patch=True) for pl in patch_lines]
            standard_line_number = item[0][1] - item[0][0]
            added_line_number = len(GetSeparateModification(item[1])[0][0])
            skip_end_lines = len(patch_lines) - standard_line_number - added_line_number
            for i in range(skip_end_lines):
                patch_lines = patch_lines[:-1]
            head_padding, tail_padding = ConfirmDiffPadding('\n'.join(patch_lines))
            start_index = 0
            end_index = 0
            while numbered_lines[start_index][0] != item[0][0] + head_padding:
                start_index += 1
            while numbered_lines[end_index][0] != item[0][1] - tail_padding - 1:
                end_index += 1
            if numbered_lines[start_index:end_index + 1]:
                buggy_function_indent = GetMinimumIndentSpaceNumber(
                    '\n'.join([content[1] for content in numbered_lines[start_index:end_index + 1]]))
            else:
                buggy_function_indent = GetMinimumIndentSpaceNumber(numbered_lines[start_index][1])
            patch_indent = GetMinimumIndentSpaceNumber(
                '\n'.join(patch_lines[head_padding:] if tail_padding == 0 else patch_lines[head_padding:-tail_padding]),
                True)
            redundant_indent = patch_indent - buggy_function_indent
            patch_lines = RemovePatchRedundantIndent('\n'.join(patch_lines), redundant_indent).split('\n')
            prefix_lines = numbered_lines[:start_index]
            suffix_lines = numbered_lines[end_index + 1:]
            numbered_all_patch = [[0, pl] for pl in patch_lines]
            numbered_patch_lines = numbered_all_patch[head_padding:] if tail_padding == 0 else numbered_all_patch[head_padding:-tail_padding]
            numbered_lines = prefix_lines + numbered_patch_lines + suffix_lines
        expanded_lines = [line[1] for line in numbered_lines]
        return '\n'.join(expanded_lines), True
    except:
        return '\n'.join([item[1] for item in patch_info]), False


def DividePatchByLineNumber(patch_content, start_no, divide_ranges):
    patch_content_lines = patch_content.split('\n')
    line_numbers = []
    line_number = start_no
    for line in patch_content_lines:
        if line.startswith('+'):
            line_numbers.append(0)
        else:
            line_numbers.append(line_number)
            line_number += 1
    numbered_lines = [[no, line] for no, line in zip(line_numbers, patch_content_lines)]
    result = []
    for divide_range in divide_ranges:
        assert divide_range[0] <= divide_range[1] and divide_range[0] <= max(line_numbers) and divide_range[
            1] >= start_no
        start_index = 0
        end_index = 0
        if divide_range[0] > start_no:
            while numbered_lines[start_index][0] != divide_range[0]:
                start_index += 1
        if divide_range[1] >= max(line_numbers):
            end_index = max(line_numbers)
        else:
            while numbered_lines[end_index][0] != divide_range[1]:
                end_index += 1
        divide_lines = numbered_lines[start_index:end_index + 1]
        result.append('\n'.join([line[1] for line in divide_lines]))
    return result


def ExtractFunctionName(function_text):
    pattern = r"\b(?:public|private|protected|static|final|abstract|synchronized)?\s*(?:[\w<>\[\]]+\s+)?(\w+)\s*\([^)]*\)"
    matches = re.findall(pattern, function_text)
    return matches


def AttachLineNumber(text, start_line=1, number_width=5, padding_right=True):
    def SpacePadding(number, width, if_right):
        if len(str(number)) >= width:
            return str(number)
        else:
            if if_right:
                return str(number) + '' * (width - len(str(number)))
            else:
                return '' * (width - len(str(number))) + str(number)
    text_lines = text.split('\n')
    line_numbers = [i for i in range(start_line, start_line + len(text_lines))]
    result_lines = []
    for i in range(len(text_lines)):
        result_lines.append(SpacePadding(line_numbers[i], number_width, padding_right) + ' ' + text_lines[i])
    return '\n'.join(result_lines)


def FocusOnTargetSlice(text, target, context_line=3, head_reserve=0):
    lines = text.strip().split('\n')
    if lines[0].split(' ')[0].isdigit():
        start_line = int(lines[0].split(' ')[0])
    else:
        start_line = 0
    end_line = len(lines) - 1
    reserve_range = [[0, head_reserve - 1]]
    reserve_point = [(item - start_line) for item in target]
    reserve_point.sort()
    for point in reserve_point:
        left = max(0, point - context_line)
        right = point + context_line
        if left < reserve_range[-1][0]:
            reserve_range[-1][0] = left
            if right > reserve_range[-1][1]:
                reserve_range[-1][1] = right
        elif reserve_range[-1][0] <= left <= reserve_range[-1][1]:
            if right > reserve_range[-1][1]:
                reserve_range[-1][1] = right
        else:
            reserve_range.append([left, right])
    result_hunks = []
    for item in reserve_range:
        if item[0] > item[1]:
            continue
        result_hunks.append('\n'.join(lines[item[0]:item[1] + 1]))
    result = "\n......\n".join(result_hunks)
    if reserve_range[-1][1] < end_line:
        result += "\n......"
    return result


def ExtractPatchContentAPPTStyle(patch_path_list):
    buggy_text_list = []
    patch_text_list = []
    for patch_path in patch_path_list:
        with open(patch_path, 'r') as f:
            patch_lines = f.readlines()
        buggy_text, patch_text = ProcessPatchContentAPPTStyle(patch_lines)
        buggy_text_list.append(buggy_text)
        patch_text_list.append(patch_text)
    return buggy_text_list, patch_text_list


def ProcessPatchContentAPPTStyle(patch_content_lines):
    diff_label = False
    end_of_comment = True
    p = r"([^\w_])"
    buggy_program = []
    patch_program = []
    for line in patch_content_lines:
        line = line.strip()
        if "*/" in line:
            end_of_comment = True
            continue
        if not end_of_comment:
            continue
        if line != '':
            if line.startswith("diff"):
                diff_label = True
            elif line.startswith("@@"):
                diff_label = False
            elif '/*' in line:
                end_of_comment = False
                continue
            elif diff_label or line.startswith("+++") or line.startswith("---"):
                continue
            elif line.startswith("+"):
                if line[1:].strip().startswith('//'):
                    continue
                line = re.split(pattern=p, string=line[1:].strip())
                line = [x.strip() for x in line]
                while '' in line:
                    line.remove('')
                clean_line = ' '.join(line).strip()
                patch_program.append(clean_line)
            elif line.startswith("-"):
                if line[1:].strip().startswith('//'):
                    continue
                line = re.split(pattern=p, string=line[1:].strip())
                line = [x.strip() for x in line]
                while '' in line:
                    line.remove('')
                clean_line = ' '.join(line).strip()
                buggy_program.append(clean_line)
            else:
                line = re.split(pattern=p, string=line.strip())
                line = [x.strip() for x in line]
                while '' in line:
                    line.remove('')
                clean_line = ' '.join(line).strip()
                buggy_program.append(clean_line)
                patch_program.append(clean_line)
    buggy_text = " ".join(buggy_program)
    patch_text = " ".join(patch_program)
    return buggy_text, patch_text


def ExtractPatchContentTian20Style(patch_path_list):
    buggy_text_list = []
    patch_text_list = []
    for patch_path in patch_path_list:
        with open(patch_path, 'r') as f:
            patch_lines = f.readlines()
        diff_label = False
        end_of_comment = True
        p = r"([^\w_])"
        buggy_parts = []
        patch_parts = []
        for line in patch_lines:
            line = line.strip()
            if "*/" in line:
                end_of_comment = True
                continue
            if not end_of_comment:
                continue
            if line != '':
                if line.startswith("diff"):
                    diff_label = True
                elif line.startswith("@@"):
                    diff_label = False
                elif '/*' in line:
                    end_of_comment = False
                    continue
                elif diff_label or line.startswith("+++") or line.startswith("---"):
                    continue
                elif line.startswith("+"):
                    if line[1:].strip().startswith('//'):
                        continue
                    line = re.split(pattern=p, string=line[1:].strip())
                    line = [x.strip() for x in line]
                    while '' in line:
                        line.remove('')
                    clean_line = ' '.join(line).strip()
                    patch_parts.append(clean_line)
                elif line.startswith("-"):
                    if line[1:].strip().startswith('//'):
                        continue
                    line = re.split(pattern=p, string=line[1:].strip())
                    line = [x.strip() for x in line]
                    while '' in line:
                        line.remove('')
                    clean_line = ' '.join(line).strip()
                    buggy_parts.append(clean_line)
        buggy_text = " ".join(buggy_parts)
        patch_text = " ".join(patch_parts)
        buggy_text_list.append(buggy_text)
        patch_text_list.append(patch_text)
    return buggy_text_list, patch_text_list


def GetFileModifiedPosition(patch_text):
    pattern = r"\b\d+\b"
    patch_lines = patch_text.split('\n')
    results = []
    file_indicator = -1
    line_counter_buggy = 0
    line_counter_patched = 0
    start_buggy = -1
    start_patched = -1
    for i in range(len(patch_lines)):
        line = patch_lines[i]
        if line.startswith("+++"):
            file_indicator += 1
            assert len(results) == file_indicator
            results.append([[], []])
        elif line.startswith("@@"):
            match = re.findall(pattern, line)
            start_buggy = int(match[0])
            start_patched = int(match[2])
            line_counter_buggy = 0
            line_counter_patched = 0
        elif line.startswith("+"):
            results[file_indicator][0].append(start_patched + line_counter_patched)
            line_counter_patched += 1
        elif line.startswith("-") and not line.startswith("---"):
            results[file_indicator][1].append(start_buggy + line_counter_buggy)
            line_counter_buggy += 1
        else:
            line_counter_buggy += 1
            line_counter_patched += 1
    return results


def FormatPlainPatchText(text):
    p = r"([^\w_])"
    result_lines = []
    for line in text.split('\n'):
        line = line.strip()
        line = re.split(pattern=p, string=line.strip())
        line = [x.strip() for x in line]
        while '' in line:
            line.remove('')
        clean_line = ' '.join(line).strip()
        result_lines.append(clean_line)
    return ' '.join(result_lines)


def LineNumbersToRange(line_numbers):
    if not line_numbers:
        return []

    ranges = []
    start = line_numbers[0]
    end = line_numbers[0]

    for num in line_numbers[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append([start, end])
            start = num
            end = num

    ranges.append([start, end])

    return ranges
