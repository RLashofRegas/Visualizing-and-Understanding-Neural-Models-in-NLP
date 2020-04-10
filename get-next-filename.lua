local function scandir(directory)
    local index = 0
    local fileNames = {}
    local dirListFile = io.popen('ls -a "'..directory..'"')
    for fileName in dirListFile:lines() do
        index = index + 1
        fileNames[index] = fileName
    end
    dirListFile:close()
    return fileNames
end

local function getNextFilename(fileNameBase, fileNameExtension, filesDir, filenameIntLength)
    -- inialize vars
    local matchingFiles = {}
    local index = 0
    
    -- get files from directory
    local fileNames = scandir(filesDir)
    
    -- get files that match fileNameBase
    for fileIndex, fileName in pairs(fileNames) do
        if fileName:sub(1, #fileNameBase) == fileNameBase then
            index = index + 1
            matchingFiles[index] = fileName
            print(fileName)
        end
    end
    
    -- sort matched file fileNames
    table.sort(matchingFiles)

    -- get last file fileName and it's corresponding number
    local lastFileName = matchingFiles[#matchingFiles]
    print(lastFileName)
    local integerStart = #fileNameBase + 2
    local lastInteger = lastFileName:sub(integerStart, integerStart + filenameIntLength)
    print(lastInteger)
    local nextInteger = tonumber(lastInteger) + 1

    -- format nextInteger
    local formatString = '%0'..filenameIntLength..'d'
    local nextIntegerString = string.format(formatString, nextInteger)

    return fileNameBase.."-"..nextIntegerString..fileNameExtension
end

return getNextFilename