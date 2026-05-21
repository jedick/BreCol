-- inline_html_tables.lua
--
-- Pandoc Lua filter that replaces a Markdown paragraph containing a link to a
-- local .html file with the parsed contents of that file. Intended for the
-- manuscript pattern:
--
--   [Table 3 data](table3_tetramer.html "Caption text shown under Table 3.")
--
-- The link target is resolved relative to Pandoc's current working directory
-- (the Makefile cds into manuscript/ before invoking pandoc). The file's HTML
-- is parsed via pandoc.read so the result renders natively in every output
-- format (LaTeX tabular for PDF, <table> for HTML, GFM table for Jekyll).
--
-- If the Markdown link has a title attribute (the quoted string after the URL),
-- the title is parsed as inline Markdown and attached as the Table's caption,
-- which makes Pandoc emit a numbered "Table N:" prefix in PDF and a <caption>
-- in HTML. Captions already present in the source HTML (e.g. a <caption> tag
-- inside the file) are left untouched.

local function is_local_html_link(target)
  if not target or target == '' then
    return false
  end
  if target:match('^%a[%w+.-]*://') or target:match('^//') or target:match('^#') then
    return false
  end
  return target:lower():match('%.html?$') ~= nil
end

local function read_file(path)
  local f = io.open(path, 'rb')
  if not f then
    return nil
  end
  local content = f:read('*all')
  f:close()
  return content
end

local function caption_is_empty(caption)
  if caption == nil then
    return true
  end
  local long = caption.long
  if long == nil or #long == 0 then
    return true
  end
  for _, block in ipairs(long) do
    if block.content and #block.content > 0 then
      return false
    end
  end
  return true
end

local function caption_from_title(title)
  local ok, doc = pcall(pandoc.read, title, 'markdown')
  if not ok or not doc or #doc.blocks == 0 then
    return nil
  end
  local inlines = pandoc.utils.blocks_to_inlines(doc.blocks)
  return pandoc.Caption({ pandoc.Plain(inlines) })
end

local function apply_caption_to_tables(blocks, caption)
  for _, block in ipairs(blocks) do
    if block.t == 'Table' and caption_is_empty(block.caption) then
      block.caption = caption
      return true
    end
  end
  return false
end

-- Convert literal "[@key]" text into pandoc Cite elements by re-parsing it as
-- Markdown. Pandoc's HTML reader keeps citation syntax as plain text, so
-- citations baked into the inlined HTML tables (e.g. helpers/table8_*.py)
-- otherwise reach citeproc as Str elements and never get resolved.
local function citation_inlines_from_text(text)
  if not text:find('%[@') then return nil end
  local ok, doc = pcall(pandoc.read, text, 'markdown')
  if not ok or not doc or #doc.blocks == 0 then return nil end
  local first = doc.blocks[1]
  if first.t ~= 'Para' and first.t ~= 'Plain' then return nil end
  for _, inline in ipairs(first.content) do
    if inline.t == 'Cite' then
      return first.content
    end
  end
  return nil
end

local citation_walker = {
  Str = function(elem)
    local inlines = citation_inlines_from_text(elem.text)
    if inlines then return inlines end
    return nil
  end,
}

local function convert_citations_in_blocks(blocks)
  -- Pandoc filters returned from Para are not re-traversed by the outer pass,
  -- so we walk the inlined-table blocks here before yielding them.
  local container = pandoc.Div(blocks)
  local walked = pandoc.walk_block(container, citation_walker)
  return walked.content
end

function Para(elem)
  for _, inline in ipairs(elem.content) do
    if inline.t == 'Link' and is_local_html_link(inline.target) then
      local content = read_file(inline.target)
      if not content then
        io.stderr:write(
          'inline_html_tables.lua: could not open ' .. inline.target ..
          ' (cwd must contain the file); leaving link unchanged\n')
        return nil
      end
      local ok, doc = pcall(pandoc.read, content, 'html')
      if not ok or not doc then
        io.stderr:write(
          'inline_html_tables.lua: failed to parse ' .. inline.target ..
          ' as HTML; leaving link unchanged\n')
        return nil
      end
      if inline.title and inline.title ~= '' then
        local caption = caption_from_title(inline.title)
        if caption then
          apply_caption_to_tables(doc.blocks, caption)
        end
      end
      return convert_citations_in_blocks(doc.blocks)
    end
  end
  return nil
end
