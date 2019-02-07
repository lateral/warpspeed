# -*- coding: utf-8 -*-

# words consist of 2 or more alphanumeric characters where there first is not
# numeric and not an underscore alphanumeric is defined in the unicode sense.
# here is a list of the unicode characters with codepoints 300 and below that
# match this pattern:
# ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzª²³µ¹º¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒ
# ÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤ
# ĥĦħĨĩĪī ...
# includes also all of greek, arabic, etc.
TOKEN_PATTERN = r'(?u)\b[^(\W|\d|_)]{1,}\w+\b'

ID_PATTERN = r'[a-zA-Z0-9\-_]+'  # ids permitted in URL slugs
