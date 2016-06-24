require 'paths'
require 'hdf5'
require 'image'

local DataGen = torch.class 'DataGen'

imgDir = '/fast_data3/knee/lua/images/playImages'

-- label classes: 
-- array(['astronaut', 'aurora', 'black', 'city', 'none', 'stars', 'unknown']

function DataGen:__init(path, trainProportion)
  -- What is nSamples????
  self.rootPath = path or imgDir
  self.nTotal = 0
  -- path would be something like root_dir data2/knee/
  -- path must start with a '/' to ensure absolute path
  local imgFile = hdf5.open(paths.concat(self.rootPath, 'hdf5s/trainImg.h5'), 'r')
  -- print (paths.concat(path, 'test/imagehdf5/3Dpatfinal.h5'))
  local imgData = imgFile:read('/imgs')
  self.nTotal = imgData:dataspaceSize()[1] -- Z dimension, total Z slices
  imgFile:close()

  -- Now, let us define a 80-20 split or user defined split to select the
  -- training set.
  self.trainProportion = math.min(0.8, trainProportion or 0.8) -- to ensure train + test =1
  self.testProportion = 0.25 * self.trainProportion -- 0.25*0.8 = 0.2
  self.trainSize = math.floor(self.trainProportion * self.nTotal)
  self.testSize = math.floor(self.testProportion * self.nTotal)
  self.batchSize = 1

end

function DataGen:commonGen(trainOrValOrTest)
  local imgFile = hdf5.open(paths.concat(self.rootPath, 'hdf5s/trainImg.h5'), 'r')
  local imgData = imgFile:read('/imgs')

  local dim = imgFile:read('/imgs'):dataspaceSize() -- dim[1], dim[2], dim[3]
  local trainSize = self.trainSize
  local testSize = self.testSize

  print('dimension of image file : ', dim)

  -- add 'validation' as well later
  if trainOrValOrTest == 'train' then
    startIndex = 1
    lastIndex = trainSize
    print("Generating training set of size:", trainSize)
  elseif trainOrValOrTest == 'test' then
    -- startIndex = trainSize + 1
    -- lastIndex = trainSize + testSize
    startIndex = 1
    lastIndex = testSize
    print("Generating test set of size:", testSize)
  else
    error ("invalid type")
  end


  local allIndices   = torch.range(startIndex, lastIndex) -- take all desired indices
  local batches      = allIndices:split(self.batchSize) -- make batches of desired size from the indices
  local randomAccess = torch.randperm(#batches):long() -- make an array of random permutation of size of bathces

  -- randomAccess = torch.range(startIndex, lastIndex)

  local i = startIndex
  local function generator()
    if i <= lastIndex then  -- trainSize + testSize for genTest
      local currBatch = batches[randomAccess[i - startIndex + 1]] --take a random batch

      local first = currBatch[1] -- start1 for partial
      local last  = currBatch[-1] -- end1 for partial , same in case of batchSize =1

      local imgData = imgFile:read('/imgs') -- needs to go inside generator, else always open

      local image = imgData:partial({first, last}, {1, dim[2]}, {1,dim[3]}, {1,dim[4]}) -- indexing starts with 1 in lua
      image = image:float();

      if i == lastIndex then
        imgFile:close()
      end

      i = i + 1 -- For 10 examples, go from 1 to 10. Close file after 10th example
      -- print(image:size())
      return image
    end -- end if i < trainSize
  end -- end local function generator

  return generator
end
