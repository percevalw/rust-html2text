//! Convert HTML to text formats.
//!
//! This crate renders HTML into a text format, wrapped to a specified width.
//! This can either be plain text or with extra annotations to (for example)
//! show in a terminal which supports colours.
//!
//! # Examples
//!
//! ```rust
//! # use html2text::from_read;
//! let html = b"
//!        <ul>
//!          <li>Item one</li>
//!          <li>Item two</li>
//!          <li>Item three</li>
//!        </ul>";
//! assert_eq!(from_read(&html[..], 20).unwrap(),
//!            "\
//! * Item one
//! * Item two
//! * Item three
//! ");
//! ```
//! A couple of simple demonstration programs are included as examples:
//!
//! ### html2text
//!
//! The simplest example uses `from_read` to convert HTML on stdin into plain
//! text:
//!
//! ```sh
//! $ cargo run --example html2text < foo.html
//! [...]
//! ```
//!
//! ### html2term
//!
//! A very simple example of using the rich interface (`from_read_rich`) for a
//! slightly interactive console HTML viewer is provided as `html2term`.
//!
//! ```sh
//! $ cargo run --example html2term foo.html
//! [...]
//! ```
//!
//! Note that this example takes the HTML file as a parameter so that it can
//! read keys from stdin.
//!

#![deny(missing_docs)]

// Check code in README.md
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadMe;

#[macro_use]
extern crate html5ever;

#[macro_use]
mod macros;

#[cfg(feature = "css")]
pub mod css;
pub mod render;

use render::text_renderer::{
    RenderLine, RenderOptions, RichAnnotation, SubRenderer, TaggedLine, TextRenderer,
};
use render::{Renderer, RichDecorator, TextDecorator, Case};
use std::borrow::Cow;

use html5ever::driver::ParseOpts;
use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use html5ever::tree_builder::TreeBuilderOpts;
mod markup5ever_rcdom;
pub use markup5ever_rcdom::RcDom;
use markup5ever_rcdom::{
    Handle,
    NodeData::{Comment, Document, Element},
};
use std::cell::Cell;
use std::cmp::{max, min};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use unicode_width::UnicodeWidthStr;

use std::io;
use std::io::Write;
use std::iter::{once, repeat};

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub(crate) enum WhiteSpace {
    #[default]
    Normal,
    // NoWrap,
    Pre,
    #[allow(unused)]
    PreWrap,
    // PreLine,
    // BreakSpaces,
}

impl WhiteSpace {
    pub fn preserve_whitespace(&self) -> bool {
        match self {
            WhiteSpace::Normal => false,
            WhiteSpace::Pre | WhiteSpace::PreWrap => true,
        }
    }
    pub fn do_wrap(&self) -> bool {
        match self {
            WhiteSpace::Normal | WhiteSpace::PreWrap => true,
            WhiteSpace::Pre => false,
        }
    }
}

/// An RGB colour value
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Colour {
    /// Red value
    pub r: u8,
    /// Green value
    pub g: u8,
    /// Blue value
    pub b: u8,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, PartialOrd)]
pub(crate) enum StyleOrigin {
    #[default]
    None,
    Agent,
    #[allow(unused)]
    User,
    #[allow(unused)]
    Author,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub(crate) struct Specificity {
    inline: bool,
    id: u16,
    class: u16,
    typ: u16,
}

impl Specificity {
    #[cfg(feature = "css")]
    fn inline() -> Self {
        Specificity {
            inline: true,
            id: 0,
            class: 0,
            typ: 0,
        }
    }
}

impl PartialOrd for Specificity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.inline.partial_cmp(&other.inline) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.id.partial_cmp(&other.id) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.class.partial_cmp(&other.class) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.typ.partial_cmp(&other.typ)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WithSpec<T> {
    val: Option<T>,
    origin: StyleOrigin,
    specificity: Specificity,
    important: bool,
}
impl<T: Copy + Clone> WithSpec<T> {
    pub(crate) fn maybe_update(
        &mut self,
        important: bool,
        origin: StyleOrigin,
        specificity: Specificity,
        val: T,
    ) {
        if self.val.is_some() {
            // We already have a value, so need to check.
            if self.important && !important {
                // important takes priority over not important.
                return;
            }
            // importance is the same.  Next is checking the origin.
            {
                use StyleOrigin::*;
                match (self.origin, origin) {
                    (Agent, Agent) | (User, User) | (Author, Author) => {
                        // They're the same so continue the comparison
                    }
                    (mine, theirs) => {
                        if (important && theirs > mine) || (!important && mine > theirs) {
                            return;
                        }
                    }
                }
            }
            // We're now from the same origin an importance
            if specificity < self.specificity {
                return;
            }
        }
        self.val = Some(val);
        self.origin = origin;
        self.specificity = specificity;
        self.important = important;
    }

    pub fn val(&self) -> Option<T> {
        self.val
    }
}

impl<T: Copy + Clone> Default for WithSpec<T> {
    fn default() -> Self {
        WithSpec {
            val: None,
            origin: StyleOrigin::None,
            specificity: Default::default(),
            important: false,
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct ComputedStyle {
    #[cfg(feature = "css")]
    /// The computed foreground colour, if any
    pub(crate) colour: WithSpec<Colour>,
    #[cfg(feature = "css")]
    /// The computed background colour, if any
    pub(crate) bg_colour: WithSpec<Colour>,
    #[cfg(feature = "css")]
    /// If set, indicates whether `display: none` or something equivalent applies
    pub(crate) display_none: WithSpec<bool>,
    /// The CSS white-space property
    pub(crate) white_space: WithSpec<WhiteSpace>,

    /// A non-CSS flag indicating we're inside a <pre>.
    pub(crate) internal_pre: bool,
}

/// Errors from reading or rendering HTML
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// The output width was too narrow to render to.
    #[error("Output width not wide enough.")]
    TooNarrow,
    /// CSS parse error
    #[error("Invalid CSS")]
    CssParseError,
    /// An general error was encountered.
    #[error("Unknown failure")]
    Fail,
    /// An I/O error
    #[error("I/O error")]
    IoError(#[from] io::Error),
}

impl PartialEq for Error {
    fn eq(&self, other: &Error) -> bool {
        use Error::*;
        match (self, other) {
            (TooNarrow, TooNarrow) => true,
            #[cfg(feature = "css")]
            (CssParseError, CssParseError) => true,
            (Fail, Fail) => true,
            _ => false,
        }
    }
}

impl Eq for Error {}

type Result<T> = std::result::Result<T, Error>;

const MIN_WIDTH: usize = 3;

/// Size information/estimate
#[derive(Debug, Copy, Clone, Default)]
struct SizeEstimate {
    size: usize,      // Rough overall size
    min_width: usize, // The narrowest possible

    // The use is specific to the node type.
    prefix_size: usize,
}

impl SizeEstimate {
    /// Combine two estimates into one (add size and take the largest
    /// min width)
    fn add(self, other: SizeEstimate) -> SizeEstimate {
        let min_width = max(self.min_width, other.min_width);
        SizeEstimate {
            size: self.size + other.size,
            min_width,
            prefix_size: 0,
        }
    }
    /// Combine two estimates into one which need to be side by side.
    /// The min widths are added.
    fn add_hor(self, other: SizeEstimate) -> SizeEstimate {
        SizeEstimate {
            size: self.size + other.size,
            min_width: self.min_width + other.min_width,
            prefix_size: 0,
        }
    }

    /// Combine two estimates into one (take max of each)
    fn max(self, other: SizeEstimate) -> SizeEstimate {
        SizeEstimate {
            size: max(self.size, other.size),
            min_width: max(self.min_width, other.min_width),
            prefix_size: 0,
        }
    }
}

#[derive(Clone, Debug)]
/// Render tree table cell
struct RenderTableCell {
    colspan: usize,
    content: Vec<RenderNode>,
    size_estimate: Cell<Option<SizeEstimate>>,
    col_width: Option<usize>, // Actual width to use
    style: ComputedStyle,
}

impl RenderTableCell {
    /// Calculate or return the estimate size of the cell
    fn get_size_estimate(&self) -> SizeEstimate {
        let Some(size) = self.size_estimate.get() else
        {
            let size = split_and_calculate_max_size(self.content.iter(), |n| n.get_size_estimate());
            self.size_estimate.set(Some(size));
            return size;
        };
        size
    }
}

#[derive(Clone, Debug)]
/// Render tree table row
struct RenderTableRow {
    cells: Vec<RenderTableCell>,
    col_sizes: Option<Vec<usize>>,
    style: ComputedStyle,
}

impl RenderTableRow {
    /// Return a mutable iterator over the cells.
    fn cells(&self) -> std::slice::Iter<RenderTableCell> {
        self.cells.iter()
    }
    /// Return a mutable iterator over the cells.
    fn cells_mut(&mut self) -> std::slice::IterMut<RenderTableCell> {
        self.cells.iter_mut()
    }
    /// Count the number of cells in the row.
    /// Takes into account colspan.
    fn num_cells(&self) -> usize {
        self.cells.iter().map(|cell| cell.colspan.max(1)).sum()
    }

    /// Return the contained cells as RenderNodes, annotated with their
    /// widths if available.  Skips cells with no width allocated.
    fn into_cells(self, vertical: bool) -> Vec<RenderNode> {
        let mut result = Vec::new();
        let mut colno = 0;
        let col_sizes = self.col_sizes.unwrap();
        for mut cell in self.cells {
            let colspan = cell.colspan;
            let col_width = if vertical {
                col_sizes[colno]
            } else {
                col_sizes[colno..colno + cell.colspan].iter().sum::<usize>()
            };
            // Skip any zero-width columns
            if col_width > 0 {
                cell.col_width = Some(col_width + cell.colspan - 1);
                let style = cell.style;
                result.push(RenderNode::new_styled(
                    RenderNodeInfo::TableCell(cell),
                    style,
                ));
            }
            colno += colspan;
        }
        result
    }
}

#[derive(Clone, Debug)]
/// A representation of a table render tree with metadata.
struct RenderTable {
    rows: Vec<RenderTableRow>,
    num_columns: usize,
    size_estimate: Cell<Option<SizeEstimate>>,
}

impl RenderTable {
    /// Create a new RenderTable with the given rows
    fn new(mut rows: Vec<RenderTableRow>) -> RenderTable {
        // We later on want to allocate a vector sized by the column count,
        // but occasionally we see something like colspan="1000000000".  We
        // handle this by remapping the column ids to the smallest values
        // possible.
        //
        // Tables with no explicit colspan will be unchanged, but if there
        // are multiple columns each covered by a single <td> on every row,
        // they will be collapsed into a single column.  For example:
        //
        //    <td><td colspan=1000><td>
        //    <td colspan=1000><td><td>
        //
        //  becomes the equivalent:
        //    <td><td colspan=2><td>
        //    <td colspan=2><td><td>

        // This will include 0 and the index after the last colspan.
        let mut col_positions = BTreeSet::new();
        col_positions.insert(0);
        for row in &rows {
            let mut col = 0;
            for cell in row.cells() {
                col += cell.colspan;
                col_positions.insert(col);
            }
        }

        let colmap: HashMap<_, _> = col_positions
            .into_iter()
            .enumerate()
            .map(|(i, pos)| (pos, i))
            .collect();

        for row in &mut rows {
            let mut pos = 0;
            let mut mapped_pos = 0;
            for cell in row.cells_mut() {
                let nextpos = pos + cell.colspan.max(1);
                let next_mapped_pos = *colmap.get(&nextpos).unwrap();
                cell.colspan = next_mapped_pos - mapped_pos;
                pos = nextpos;
                mapped_pos = next_mapped_pos;
            }
        }

        let num_columns = rows.iter().map(|r| r.num_cells()).max().unwrap_or(0);
        RenderTable {
            rows,
            num_columns,
            size_estimate: Cell::new(None),
        }
    }

    /// Return an iterator over the rows.
    fn rows(&self) -> std::slice::Iter<RenderTableRow> {
        self.rows.iter()
    }

    /// Consume this and return a `Vec<RenderNode>` containing the children;
    /// the children know the column sizes required.
    fn into_rows(self, col_sizes: Vec<usize>, vert: bool) -> Vec<RenderNode> {
        self.rows
            .into_iter()
            .map(|mut tr| {
                tr.col_sizes = Some(col_sizes.clone());
                let style = tr.style;
                RenderNode::new_styled(RenderNodeInfo::TableRow(tr, vert), style)
            })
            .collect()
    }

    fn calc_size_estimate(&self, _context: &HtmlContext) -> SizeEstimate {
        if self.num_columns == 0 {
            let result = SizeEstimate {
                size: 0,
                min_width: 0,
                prefix_size: 0,
            };
            self.size_estimate.set(Some(result));
            return result;
        }
        let mut sizes: Vec<SizeEstimate> = vec![Default::default(); self.num_columns];

        // For now, a simple estimate based on adding up sub-parts.
        for row in self.rows() {
            let mut colno = 0usize;
            for cell in row.cells() {
                let cellsize = cell.get_size_estimate();
                // println!("Cellsize {:?} for cell {:?}", cellsize, cell);
                for colnum in 0..cell.colspan {
                    // let border = if (colnum + colno > 0) { 1 } else { 0 };
                    // println!("Cellsize {:?}, Colnum: {:?}, Border: {} for cell {:?}", cellsize, colnum, border, cell);
                    sizes[colno + colnum].size = max(
                        sizes[colno + colnum].size,
                        cellsize.size / cell.colspan, //+ border,
                    );
                    sizes[colno + colnum].min_width = max(
                        sizes[colno + colnum].min_width,
                        cellsize.min_width / cell.colspan, //+ border,
                    );
                }
                colno += cell.colspan;
            }
        }
        // println!("Sizes: {:?}", sizes);
        let size = sizes.iter().map(|s| s.size).sum(); // Include borders?
        let min_width = sizes.iter().map(|s| s.min_width).sum::<usize>() + self.num_columns - 1;
        let result = SizeEstimate {
            size,
            min_width,
            prefix_size: 0,
        };
        self.size_estimate.set(Some(result));
        result
    }
}

fn split_and_calculate_max_size<'a, I>(
    nodes: I,
    recurse: impl Fn(&RenderNode) -> SizeEstimate,
) -> SizeEstimate
where
    I: Iterator<Item=&'a RenderNode>,
{
    use RenderNodeInfo::*;
    let mut max_size_estimate = SizeEstimate::default();
    let mut current_chunk = Vec::new();

    for node in nodes {
        if let Break = node.info {
            // Compute the size for the current chunk and update max_size_estimate
            if !current_chunk.is_empty() {
                let chunk_size = current_chunk
                    .iter()
                    .map(|n| recurse(*n))
                    .fold(Default::default(), SizeEstimate::add);
                max_size_estimate = max_size_estimate.max(chunk_size);
                // println!("  -------------------");
                // println!("  current_chunk: {:?}", current_chunk);
                // println!("  max_size_estimate: {:?}", max_size_estimate);
                current_chunk.clear();
            }
        } else {
            current_chunk.push(node);
        }
    }

    // Handle the last chunk if it exists
    if !current_chunk.is_empty() {
        let chunk_size = current_chunk
            .iter()
            .map(|n| recurse(*n))
            .fold(Default::default(), SizeEstimate::add);
        max_size_estimate = max_size_estimate.max(chunk_size);
        // println!("  -------------------");
        // println!("  current_chunk: {:?}", current_chunk);
        // println!("  max_size_estimate: {:?}", max_size_estimate);
    }

    max_size_estimate
}

/// The node-specific information distilled from the DOM.
#[derive(Clone, Debug)]
#[non_exhaustive]
enum RenderNodeInfo {
    /// Some text.
    Text(String),
    /// A group of nodes collected together.
    Container(Vec<RenderNode>),
    /// A container that starts with a fragment marker
    FragmentContainer(Vec<RenderNode>),
    /// A link with contained nodes
    Link(String, Vec<RenderNode>),
    /// An emphasised region
    Em(Vec<RenderNode>),
    /// A strong region
    Strong(Vec<RenderNode>),
    /// A struck out region
    Strikeout(Vec<RenderNode>),
    /// A code region
    Code(Vec<RenderNode>),
    /// An image (src, title)
    Img(String, String),
    /// A block element with children
    Block(Vec<RenderNode>),
    /// A header (h1, h2, ...) with children
    Header(usize, Vec<RenderNode>),
    /// A Div element with children
    Div(Vec<RenderNode>),
    /// A blockquote
    BlockQuote(Vec<RenderNode>),
    /// An unordered list
    Ul(Vec<RenderNode>),
    /// An ordered list
    Ol(i64, Vec<RenderNode>),
    /// A description list (containing Dt or Dd)
    Dl(Vec<RenderNode>),
    /// A term (from a `<dl>`)
    Dt(Vec<RenderNode>),
    /// A definition (from a `<dl>`)
    Dd(Vec<RenderNode>),
    /// A line break
    Break,
    /// A table
    Table(RenderTable),
    /// A set of table rows (from either `<thead>` or `<tbody>`
    TableBody(Vec<RenderTableRow>),
    /// Table row (must only appear within a table body)
    /// If the boolean is true, then the cells are drawn vertically
    /// instead of horizontally (because of space).
    TableRow(RenderTableRow, bool),
    /// Table cell (must only appear within a table row)
    TableCell(RenderTableCell),
    /// Start of a named HTML fragment
    FragStart(String),
    /// A list item
    ListItem(Vec<RenderNode>),
    /// Superscript text
    Sup(Vec<RenderNode>),

    /// Form, not rendered
    Form,
}

/// Common fields from a node.
#[derive(Clone, Debug)]
struct RenderNode {
    size_estimate: Cell<Option<SizeEstimate>>,
    info: RenderNodeInfo,
    style: ComputedStyle,
}

impl RenderNode {
    /// Create a node from the RenderNodeInfo.
    fn new(info: RenderNodeInfo) -> RenderNode {
        RenderNode {
            size_estimate: Cell::new(None),
            info,
            style: Default::default(),
        }
    }

    /// Create a node from the RenderNodeInfo.
    fn new_styled(info: RenderNodeInfo, style: ComputedStyle) -> RenderNode {
        RenderNode {
            size_estimate: Cell::new(None),
            info,
            style,
        }
    }

    /// Get a size estimate
    fn get_size_estimate(&self) -> SizeEstimate {
        self.size_estimate.get().unwrap()
    }

    /// Calculate the size of this node.
    fn calc_size_estimate<D: TextDecorator>(
        &self,
        context: &HtmlContext,
        decorator: &D,
    ) -> SizeEstimate {
        // If it's already calculated, then just return the answer.
        if let Some(s) = self.size_estimate.get() {
            return s;
        };

        use RenderNodeInfo::*;

        let recurse = |node: &RenderNode| node.calc_size_estimate(context, decorator);

        // Otherwise, make an estimate.
        let estimate = match self.info {
            Text(ref t) | Img(_, ref t) => {
                use unicode_width::UnicodeWidthChar;
                let mut len = 0;
                let mut in_whitespace = false;
                for c in t.trim_start().chars() {
                    let is_ws = c.is_whitespace();
                    if !is_ws {
                        len += UnicodeWidthChar::width(c).unwrap_or(0);
                        // Count the preceding whitespace as one.
                        if in_whitespace {
                            len += 1;
                        }
                    }
                    in_whitespace = is_ws;
                }
                // Add one for preceding whitespace.
                if let Some(true) = t.chars().next().map(|c| c.is_whitespace()) {
                    len += 1;
                }
                if in_whitespace {
                    len += 1;
                }
                /*if let Img(_, _) = self.info {
                    len += 2;
                }*/
                // println!("Text: {:?} -> {:?}", t, len);
                SizeEstimate {
                    size: len,
                    min_width: len.min(context.min_wrap_width),
                    prefix_size: 0,
                }
            }

            Container(ref v) | FragmentContainer(ref v) | Em(ref v) | Strong(ref v) | Strikeout(ref v) | Code(ref v)
            | Block(ref v) | Dl(ref v) | Dt(ref v) | ListItem(ref v) | Sup(ref v) => {
                split_and_calculate_max_size(v.iter(), recurse)
            }
            Div(ref v) => {
                let mut size = v
                    .iter()
                    .map(recurse)
                    .fold(Default::default(), SizeEstimate::max);
                size
            }
            Link(ref _target, ref v) => v
                .iter()
                .map(recurse)
                .fold(Default::default(), SizeEstimate::add),
            // .add(SizeEstimate {
            //     size: 5,
            //     min_width: 5,
            //     prefix_size: 0,
            // }),
            Dd(ref v) | BlockQuote(ref v) | Ul(ref v) => {
                let prefix = match self.info {
                    Dd(_) => "  ".into(),
                    BlockQuote(_) => decorator.quote_prefix(),
                    Ul(_) => decorator.unordered_item_prefix(),
                    _ => unreachable!(),
                };
                let prefix_width = UnicodeWidthStr::width(prefix.as_str());
                let mut size = v
                    .iter()
                    .map(recurse)
                    .fold(Default::default(), SizeEstimate::max)
                    .add_hor(SizeEstimate {
                        size: prefix_width,
                        min_width: prefix_width,
                        prefix_size: 0,
                    });
                size.prefix_size = prefix_width;
                size
            }
            Ol(i, ref v) => {
                let prefix_size = calc_ol_prefix_size(i, v.len(), decorator);
                let mut result = v
                    .iter()
                    .map(recurse)
                    .fold(Default::default(), SizeEstimate::max)
                    .add_hor(SizeEstimate {
                        size: prefix_size,
                        min_width: prefix_size,
                        prefix_size: 0,
                    });
                result.prefix_size = prefix_size;
                result
            }
            Header(level, ref v) => {
                let prefix_size = decorator.header_prefix(level).len();
                let mut size = v
                    .iter()
                    .map(recurse)
                    .fold(Default::default(), SizeEstimate::add)
                    .add_hor(SizeEstimate {
                        size: prefix_size,
                        min_width: prefix_size,
                        prefix_size: 0,
                    });
                size.prefix_size = prefix_size;
                size
            }
            Break => SizeEstimate {
                size: 0,
                min_width: 0,
                prefix_size: 0,
            },
            Table(ref t) => t.calc_size_estimate(context),
            TableRow(..) | TableBody(_) | TableCell(_) => unimplemented!(),
            FragStart(_) => Default::default(),
            Form => Default::default(),
        };
        self.size_estimate.set(Some(estimate));
        estimate
    }

    /// Return true if this node is definitely empty.  This is used to quickly
    /// remove e.g. links with no anchor text in most cases, but can't recurse
    /// and look more deeply.
    fn is_shallow_empty(&self) -> bool {
        use RenderNodeInfo::*;

        // Otherwise, make an estimate.
        match self.info {
            Text(ref t) | Img(_, ref t) => {
                let len = t.trim().len();
                len == 0
            }

            Container(ref v)
            | FragmentContainer(ref v)
            | Link(_, ref v)
            | Em(ref v)
            | Strong(ref v)
            | Strikeout(ref v)
            | Code(ref v)
            | Block(ref v)
            | ListItem(ref v)
            | Div(ref v)
            | BlockQuote(ref v)
            | Dl(ref v)
            | Dt(ref v)
            | Dd(ref v)
            | Ul(ref v)
            | Ol(_, ref v)
            | Sup(ref v) => v.is_empty(),
            Header(_level, ref v) => v.is_empty(),
            Break => true,
            Table(ref _t) => false,
            TableRow(..) | TableBody(_) | TableCell(_) => false,
            FragStart(_) => true,
            Form => false,
        }
    }

    fn write_container(
        &self,
        name: &str,
        items: &[RenderNode],
        f: &mut std::fmt::Formatter,
        indent: usize,
    ) -> std::prelude::v1::Result<(), std::fmt::Error> {
        writeln!(f, "{:indent$}{name}:", "")?;
        for item in items {
            item.write_self(f, indent + 1)?;
        }
        Ok(())
    }
    fn write_style(
        f: &mut std::fmt::Formatter,
        indent: usize,
        style: &ComputedStyle,
    ) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "{:indent$}[Style:", "")?;

        #[cfg(feature = "css")]
        {
            if let Some(col) = style.colour.val() {
                write!(f, " colour={:?}", col)?;
            }
            if let Some(col) = style.bg_colour.val() {
                write!(f, " bg_colour={:?}", col)?;
            }
            if let Some(val) = style.display_none.val() {
                write!(f, " disp_none={:?}", val)?;
            }
        }
        if let Some(ws) = style.white_space.val() {
            write!(f, " white_space={:?}", ws)?;
        }
        if style.internal_pre {
            write!(f, " internal_pre")?;
        }
        writeln!(f, "")
    }
    fn write_self(
        &self,
        f: &mut std::fmt::Formatter,
        indent: usize,
    ) -> std::prelude::v1::Result<(), std::fmt::Error> {
        Self::write_style(f, indent, &self.style)?;

        match &self.info {
            RenderNodeInfo::Text(s) => writeln!(f, "{:indent$}{s:?}", "")?,
            RenderNodeInfo::Container(v) | RenderNodeInfo::FragmentContainer(v) => {
                self.write_container("Container", &v, f, indent)?;
            }
            RenderNodeInfo::Link(targ, v) => {
                self.write_container(&format!("Link({})", targ), &v, f, indent)?;
            }
            RenderNodeInfo::Em(v) => {
                self.write_container("Em", &v, f, indent)?;
            }
            RenderNodeInfo::Strong(v) => {
                self.write_container("Strong", &v, f, indent)?;
            }
            RenderNodeInfo::Strikeout(v) => {
                self.write_container("Strikeout", &v, f, indent)?;
            }
            RenderNodeInfo::Code(v) => {
                self.write_container("Code", &v, f, indent)?;
            }
            RenderNodeInfo::Img(src, title) => {
                writeln!(f, "{:indent$}Img src={:?} title={:?}:", "", src, title)?;
            }
            RenderNodeInfo::Block(v) => {
                self.write_container("Block", &v, f, indent)?;
            }
            RenderNodeInfo::Header(depth, v) => {
                self.write_container(&format!("Header({})", depth), &v, f, indent)?;
            }
            RenderNodeInfo::Div(v) => {
                self.write_container("Div", &v, f, indent)?;
            }
            RenderNodeInfo::BlockQuote(v) => {
                self.write_container("BlockQuote", &v, f, indent)?;
            }
            RenderNodeInfo::Ul(v) => {
                self.write_container("Ul", &v, f, indent)?;
            }
            RenderNodeInfo::Ol(start, v) => {
                self.write_container(&format!("Ol({})", start), &v, f, indent)?;
            }
            RenderNodeInfo::Dl(v) => {
                self.write_container("Dl", &v, f, indent)?;
            }
            RenderNodeInfo::Dt(v) => {
                self.write_container("Dt", &v, f, indent)?;
            }
            RenderNodeInfo::Dd(v) => {
                self.write_container("Dd", &v, f, indent)?;
            }
            RenderNodeInfo::Break => {
                writeln!(f, "{:indent$}Break", "", indent = indent)?;
            }
            RenderNodeInfo::Table(rows) => {
                writeln!(f, "{:indent$}Table ({} cols):", "", rows.num_columns)?;
                for rtr in &rows.rows {
                    Self::write_style(f, indent + 1, &rtr.style)?;
                    writeln!(
                        f,
                        "{:width$}Row ({} cells):",
                        "",
                        rtr.cells.len(),
                        width = indent + 1
                    )?;
                    for cell in &rtr.cells {
                        Self::write_style(f, indent + 2, &cell.style)?;
                        writeln!(
                            f,
                            "{:width$}Cell colspan={} width={:?}:",
                            "",
                            cell.colspan,
                            cell.col_width,
                            width = indent + 2
                        )?;
                        for node in &cell.content {
                            node.write_self(f, indent + 3)?;
                        }
                    }
                }
            }
            RenderNodeInfo::TableBody(_) => todo!(),
            RenderNodeInfo::TableRow(_, _) => todo!(),
            RenderNodeInfo::TableCell(_) => todo!(),
            RenderNodeInfo::FragStart(frag) => {
                writeln!(f, "{:indent$}FragStart({}):", "", frag)?;
            }
            RenderNodeInfo::ListItem(v) => {
                self.write_container("ListItem", &v, f, indent)?;
            }
            RenderNodeInfo::Sup(v) => {
                self.write_container("Sup", &v, f, indent)?;
            }
            RenderNodeInfo::Form => {
                writeln!(f, "{:indent$}Form", "", indent = indent)?;
            }
        }
        Ok(())
    }
}

fn precalc_size_estimate<'a, D: TextDecorator>(
    node: &'a RenderNode,
    context: &mut HtmlContext,
    decorator: &'a D,
) -> TreeMapResult<'a, HtmlContext, &'a RenderNode, ()> {
    use RenderNodeInfo::*;
    if node.size_estimate.get().is_some() {
        return TreeMapResult::Nothing;
    }
    match node.info {
        Text(_) | Img(_, _) | Break | FragStart(_) => {
            let _ = node.calc_size_estimate(context, decorator);
            TreeMapResult::Nothing
        }

        Container(ref v)
        | FragmentContainer(ref v)
        | Link(_, ref v)
        | Em(ref v)
        | Strong(ref v)
        | Strikeout(ref v)
        | Code(ref v)
        | Block(ref v)
        | ListItem(ref v)
        | Div(ref v)
        | BlockQuote(ref v)
        | Ul(ref v)
        | Ol(_, ref v)
        | Dl(ref v)
        | Dt(ref v)
        | Dd(ref v)
        | Sup(ref v)
        | Header(_, ref v) => TreeMapResult::PendingChildren {
            children: v.iter().collect(),
            cons: Box::new(move |context, _cs| {
                node.calc_size_estimate(context, decorator);
                Ok(None)
            }),
            prefn: None,
            postfn: None,
        },
        Table(ref t) => {
            /* Return all the indirect children which are RenderNodes. */
            let mut children = Vec::new();
            for row in &t.rows {
                for cell in &row.cells {
                    children.extend(cell.content.iter());
                }
            }
            TreeMapResult::PendingChildren {
                children,
                cons: Box::new(move |context, _cs| {
                    node.calc_size_estimate(context, decorator);
                    Ok(None)
                }),
                prefn: None,
                postfn: None,
            }
        }
        TableRow(..) | TableBody(_) | TableCell(_) => unimplemented!(),
        Form => TreeMapResult::Nothing,
    }
}

/// Make a Vec of RenderNodes from the children of a node.
fn children_to_render_nodes<T: Write>(
    handle: Handle,
    err_out: &mut T,
    context: &mut HtmlContext,
) -> Result<Vec<RenderNode>> {
    /* process children, but don't add anything */
    handle
        .children
        .borrow()
        .iter()
        .flat_map(|ch| dom_to_render_tree_with_context(ch.clone(), err_out, context).transpose())
        .collect()
}

/// Make a Vec of DtElements from the `<dt>` and `<dd>` children of a node.
fn desc_list_children_to_render_nodes<T: Write>(
    handle: Handle,
    err_out: &mut T,
    context: &mut HtmlContext,
) -> Result<Vec<RenderNode>> {
    let mut children = Vec::new();

    for child in handle.children.borrow().iter() {
        match child.data {
            Element { ref name, .. } => match name.expanded() {
                expanded_name!(html "dt") => {
                    let dt_children = children_to_render_nodes(child.clone(), err_out, context)?;
                    children.push(RenderNode::new(RenderNodeInfo::Dt(dt_children)));
                }
                expanded_name!(html "dd") => {
                    let dd_children = children_to_render_nodes(child.clone(), err_out, context)?;
                    children.push(RenderNode::new(RenderNodeInfo::Dd(dd_children)));
                }
                _ => {}
            },
            Comment { .. } => {}
            _ => {
                html_trace!("Unhandled in list: {:?}\n", child);
            }
        }
    }
    Ok(children)
}

/// Convert a table into a RenderNode
fn table_to_render_tree<'a, T: Write>(
    input: RenderInput,
    computed: ComputedStyle,
    _err_out: &mut T,
) -> TreeMapResult<'a, HtmlContext, RenderInput, RenderNode> {
    pending(input, move |_, rowset| {
        let mut rows = vec![];
        let mut num_cols = 0;
        for bodynode in rowset {
            if let RenderNodeInfo::TableBody(body) = bodynode.info {
                num_cols = num_cols.max(body.iter().map(|r| r.num_cells()).max().unwrap_or(0));
                rows.extend(body);
            } else {
                html_trace!("Found in table: {:?}", bodynode.info);
            }
        }
        if rows.is_empty() {
            None
        } else if num_cols == 1 {
            // map(
            //                     |node| RenderNode::new_styled(node.info, node.style),
            //                 )
            let cells = (
                rows
                    .into_iter()
                    .flat_map(|row: RenderTableRow| row.cells.into_iter())
                    .flat_map(|c| c.content.clone())
                    .collect()
            );
            Some(RenderNode::new_styled(
                RenderNodeInfo::Container(cells),
                computed,
            ))
        } else {
            Some(RenderNode::new_styled(
                RenderNodeInfo::Table(RenderTable::new(rows)),
                computed,
            ))
        }
    })
}

/// Add rows from a thead or tbody.
fn tbody_to_render_tree<'a, T: Write>(
    input: RenderInput,
    computed: ComputedStyle,
    _err_out: &mut T,
) -> TreeMapResult<'a, HtmlContext, RenderInput, RenderNode> {
    pending_noempty(input, move |_, rowchildren| {
        let mut rows = rowchildren
            .into_iter()
            .flat_map(|rownode| {
                if let RenderNodeInfo::TableRow(row, _) = rownode.info {
                    Some(row)
                } else {
                    html_trace!("  [[tbody child: {:?}]]", rownode);
                    None
                }
            })
            .collect::<Vec<_>>();

        // Handle colspan=0 by replacing it.
        // Get a list of (has_zero_colspan, sum_colspan)
        let num_columns = rows
            .iter()
            .map(|row| {
                row.cells()
                    // Treat the column as having colspan 1 for initial counting.
                    .map(|cell| (cell.colspan == 0, cell.colspan.max(1)))
                    .fold((false, 0), |a, b| (a.0 || b.0, a.1 + b.1))
            })
            .collect::<Vec<_>>();

        let max_columns = num_columns.iter().map(|(_, span)| span).max().unwrap_or(&1);

        for (i, &(has_zero, num_cols)) in num_columns.iter().enumerate() {
            // Note this won't be sensible if more than one column has colspan=0,
            // but that's not very well defined anyway.
            if has_zero {
                for cell in rows[i].cells_mut() {
                    if cell.colspan == 0 {
                        // +1 because we said it had 1 to start with
                        cell.colspan = max_columns - num_cols + 1;
                    }
                }
            }
        }

        Some(RenderNode::new_styled(
            RenderNodeInfo::TableBody(rows),
            computed,
        ))
    })
}

/// Convert a table row to a RenderTableRow
fn tr_to_render_tree<'a, T: Write>(
    input: RenderInput,
    computed: ComputedStyle,
    _err_out: &mut T,
) -> TreeMapResult<'a, HtmlContext, RenderInput, RenderNode> {
    pending(input, move |_, cellnodes| {
        let cells = cellnodes
            .into_iter()
            .flat_map(|cellnode| {
                if let RenderNodeInfo::TableCell(cell) = cellnode.info {
                    Some(cell)
                } else {
                    html_trace!("  [[tr child: {:?}]]", cellnode);
                    None
                }
            })
            .collect();
        Some(RenderNode::new_styled(
            RenderNodeInfo::TableRow(
                RenderTableRow {
                    cells,
                    col_sizes: None,
                    style: computed,
                },
                false,
            ),
            computed,
        ))
    })
}

/// Convert a single table cell to a render node.
fn td_to_render_tree<'a, T: Write>(
    input: RenderInput,
    computed: ComputedStyle,
    _err_out: &mut T,
) -> TreeMapResult<'a, HtmlContext, RenderInput, RenderNode> {
    let mut colspan = 1;
    if let Element { ref attrs, .. } = input.handle.data {
        for attr in attrs.borrow().iter() {
            if &attr.name.local == "colspan" {
                let v: &str = &attr.value;
                colspan = v.parse().unwrap_or(1);
            }
        }
    }
    pending(input, move |_, children| {
        Some(RenderNode::new_styled(
            RenderNodeInfo::TableCell(RenderTableCell {
                colspan,
                content: children,
                size_estimate: Cell::new(None),
                col_width: None,
                style: computed,
            }),
            computed,
        ))
    })
}

/// A reducer which combines results from mapping children into
/// the result for the current node.  Takes a context and a
/// vector of results and returns a new result (or nothing).
type ResultReducer<'a, C, R> = dyn FnOnce(&mut C, Vec<R>) -> Result<Option<R>> + 'a;

/// A closure to call before processing a child node.
type ChildPreFn<C, N> = dyn Fn(&mut C, &N) -> Result<()>;

/// A closure to call after processing a child node,
/// before adding the result to the processed results
/// vector.
type ChildPostFn<C, R> = dyn Fn(&mut C, &R) -> Result<()>;

/// The result of trying to render one node.
enum TreeMapResult<'a, C, N, R> {
    /// A completed result.
    Finished(R),
    /// Deferred completion - can be turned into a result
    /// once the vector of children are processed.
    PendingChildren {
        children: Vec<N>,
        cons: Box<ResultReducer<'a, C, R>>,
        prefn: Option<Box<ChildPreFn<C, N>>>,
        postfn: Option<Box<ChildPostFn<C, R>>>,
    },
    /// Nothing (e.g. a comment or other ignored element).
    Nothing,
}

fn tree_map_reduce<'a, C, N, R, M>(
    context: &mut C,
    top: N,
    mut process_node: M,
) -> Result<Option<R>>
where
    M: FnMut(&mut C, N) -> Result<TreeMapResult<'a, C, N, R>>,
{
    /// A node partially decoded, waiting for its children to
    /// be processed.
    struct PendingNode<'a, C, R, N> {
        /// How to make the node once finished
        construct: Box<ResultReducer<'a, C, R>>,
        /// Called before processing each child
        prefn: Option<Box<ChildPreFn<C, N>>>,
        /// Called after processing each child
        postfn: Option<Box<ChildPostFn<C, R>>>,
        /// Children already processed
        children: Vec<R>,
        /// Iterator of child nodes not yet processed
        to_process: std::vec::IntoIter<N>,
    }

    let mut last = PendingNode {
        // We only expect one child, which we'll just return.
        construct: Box::new(|_, mut cs| Ok(cs.pop())),
        prefn: None,
        postfn: None,
        children: Vec::new(),
        to_process: vec![top].into_iter(),
    };
    let mut pending_stack = Vec::new();
    loop {
        // Get the next child node to process
        while let Some(h) = last.to_process.next() {
            if let Some(f) = &last.prefn {
                f(context, &h)?;
            }
            match process_node(context, h)? {
                TreeMapResult::Finished(result) => {
                    if let Some(f) = &last.postfn {
                        f(context, &result)?;
                    }
                    last.children.push(result);
                }
                TreeMapResult::PendingChildren {
                    children,
                    cons,
                    prefn,
                    postfn,
                } => {
                    pending_stack.push(last);
                    last = PendingNode {
                        construct: cons,
                        prefn,
                        postfn,
                        children: Vec::new(),
                        to_process: children.into_iter(),
                    };
                }
                TreeMapResult::Nothing => {}
            };
        }
        // No more children, so finally construct the parent.
        if let Some(mut parent) = pending_stack.pop() {
            if let Some(node) = (last.construct)(context, last.children)? {
                if let Some(f) = &parent.postfn {
                    f(context, &node)?;
                }
                parent.children.push(node);
            }
            last = parent;
            continue;
        }
        // Finished the whole stack!
        break Ok((last.construct)(context, last.children)?);
    }
}

#[derive(Default, Debug)]
struct HtmlContext {
    #[cfg(feature = "css")]
    style_data: css::StyleData,
    #[cfg(feature = "css")]
    use_doc_css: bool,

    max_wrap_width: Option<usize>,
    pad_block_width: bool,
    allow_width_overflow: bool,
    min_wrap_width: usize,
    raw: bool,
    draw_borders: bool,
    wrap_links: bool,
    done: bool,
}

// Input to render tree conversion.
struct RenderInput {
    handle: Handle,
    parent_style: Rc<ComputedStyle>,
}

impl RenderInput {
    // Return the children in the right form
    fn children(&self) -> Vec<RenderInput> {
        self.handle
            .children
            .borrow()
            .iter()
            .map(|child| RenderInput {
                handle: child.clone(),
                parent_style: Rc::clone(&self.parent_style),
            })
            .collect()
    }
}

fn dom_to_render_tree_with_context<T: Write>(
    handle: Handle,
    err_out: &mut T,
    context: &mut HtmlContext,
) -> Result<Option<RenderNode>> {
    html_trace!("### dom_to_render_tree: HTML: {:?}", handle);
    #[cfg(feature = "css")]
    if context.use_doc_css {
        let mut doc_style_data = css::dom_to_stylesheet(handle.clone(), err_out)?;
        doc_style_data.merge(std::mem::take(&mut context.style_data));
        context.style_data = doc_style_data;
    }

    let parent_style = Default::default();
    let result = tree_map_reduce(
        context,
        RenderInput {
            handle,
            parent_style,
        },
        |context, input| process_dom_node(input, err_out, context),
    );

    html_trace!("### dom_to_render_tree: out= {:#?}", result);
    result
}

fn pending<F>(
    input: RenderInput,
    f: F,
) -> TreeMapResult<'static, HtmlContext, RenderInput, RenderNode>
where
    F: Fn(&mut HtmlContext, Vec<RenderNode>) -> Option<RenderNode> + 'static,
{
    TreeMapResult::PendingChildren {
        children: input.children(),
        cons: Box::new(move |ctx, children| Ok(f(ctx, children))),
        prefn: None,
        postfn: None,
    }
}

fn pending_noempty<F>(
    input: RenderInput,
    f: F,
) -> TreeMapResult<'static, HtmlContext, RenderInput, RenderNode>
where
    F: Fn(&mut HtmlContext, Vec<RenderNode>) -> Option<RenderNode> + 'static,
{
    let handle = &input.handle;
    let style = &input.parent_style;
    TreeMapResult::PendingChildren {
        children: handle
            .children
            .borrow()
            .iter()
            .map(|child| RenderInput {
                handle: child.clone(),
                parent_style: Rc::clone(style),
            })
            .collect(),
        cons: Box::new(move |ctx, children| {
            if children.is_empty() {
                Ok(None)
            } else {
                Ok(f(ctx, children))
            }
        }),
        prefn: None,
        postfn: None,
    }
}

/// Prepend a FragmentStart (or analogous) marker to an existing
/// RenderNode.
fn prepend_marker(prefix: RenderNode, mut orig: RenderNode) -> RenderNode {
    use RenderNodeInfo::*;
    html_trace!("prepend_marker({:?}, {:?})", prefix, orig);

    match orig.info {
        // For block elements such as Block and Div, we need to insert
        // the node at the front of their children array, otherwise
        // the renderer is liable to drop the fragment start marker
        // _before_ the new line indicating the end of the previous
        // paragraph.
        //
        // For Container, we do the same thing just to make the data
        // less pointlessly nested.
        Block(ref mut children)
        | ListItem(ref mut children)
        | Div(ref mut children)
        | BlockQuote(ref mut children)
        | FragmentContainer(ref mut children)
        | TableCell(RenderTableCell {
                        content: ref mut children,
                        ..
                    }) => {
            children.insert(0, prefix);
            // Now return orig, but we do that outside the match so
            // that we've given back the borrowed ref 'children'.
        }

        // For table rows and tables, push down if there's any content.
        TableRow(ref mut rrow, _) => {
            // If the row is empty, then there isn't really anything
            // to attach the fragment start to.
            if let Some(cell) = rrow.cells.first_mut() {
                cell.content.insert(0, prefix);
            }
        }

        TableBody(ref mut rows) | Table(RenderTable { ref mut rows, .. }) => {
            // If the row is empty, then there isn't really anything
            // to attach the fragment start to.
            if let Some(rrow) = rows.first_mut() {
                if let Some(cell) = rrow.cells.first_mut() {
                    cell.content.insert(0, prefix);
                }
            }
        }

        // For anything else, just make a new Container with the
        // prefix node and the original one.
        _ => {
            if !check_container(&vec![prefix.clone(), orig.clone()]) {
                return orig;
            }
            let result = RenderNode::new(FragmentContainer(vec![prefix, orig]));
            html_trace!("prepend_marker() -> {:?}", result);
            return result;
        }
    }
    html_trace!("prepend_marker() -> {:?}", &orig);
    orig
}

fn check_container(
    children: &Vec<RenderNode>,
) -> bool {
    // Check if there is a form
    return true;
    for c in children {
        match &c.info {
            RenderNodeInfo::Form => {
                break;
            }
            RenderNodeInfo::FragmentContainer(v) => {
                if !check_container(v) {
                    return false;
                }
            }
            _ => {}
        }
    }
    let form_exist = children.iter().any(|c| {
        if let RenderNodeInfo::Form = c.info {
            true
        } else {
            false
        }
    });
    return !form_exist;
}

fn split_alpha_chains(s: &&str) -> Vec<String> {
    s.split(|c: char| !c.is_alphabetic())
        .flat_map(|word| {
            let mut words = Vec::new();
            let mut current_word = String::new();
            let mut prev_is_lower = false;

            for c in word.chars() {
                if c.is_uppercase() && prev_is_lower {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                current_word.extend(c.to_lowercase());
                prev_is_lower = c.is_lowercase();
            }

            if !current_word.is_empty() {
                words.push(current_word);
            }

            words
        }).collect()
}

fn process_dom_node<T: Write>(
    input: RenderInput,
    err_out: &mut T,
    #[allow(unused)] // Used with css feature
    context: &mut HtmlContext,
) -> Result<TreeMapResult<'static, HtmlContext, RenderInput, RenderNode>> {
    use RenderNodeInfo::*;
    use TreeMapResult::*;
    if context.done {
        return Ok(Nothing);
    }

    Ok(match input.handle.clone().data {
        Document => pending(input, |_context, cs| Some(RenderNode::new(Container(cs)))),
        Comment { .. } => Nothing,
        Element {
            ref name,
            ref attrs,
            ..
        } => {
            let mut frag_from_name_attr = false;

            let attrs_ = attrs.borrow();
            // Check the first (and only the first) class attr class if it exists
            // for a fragment name.
            let class = attrs_
                .iter()
                .filter(|attr| &attr.name.local == "class" || &attr.name.local == "id" || &attr.name.local == "role") //  || (&attr.name.local).starts_with("data"))
                .map(|attr| attr.value.as_ref())
                .collect::<Vec<_>>();
            // split the class attr into words (contiguous letter words, skipping numbers)
            // and collect as a set
            // eg: "class1 class2" -> {"class", "class"}

            // Check if any of the class tokens start with "menu", "nav" or "footer"
            let mut to_skip = !(name.expanded() == expanded_name!(html "body") || name.expanded() == expanded_name!(html "html")) &&
                ((
                    class.iter().flat_map(split_alpha_chains)
                ).any(|class|
                    class.starts_with("menu")
                        || class.starts_with("nav")
                        || class.starts_with("btn")
                        || class.starts_with("button")
                        || class.starts_with("related")
                        || class == "pager"
                        || class == "banner"
                        || class == "sidebar"
                        || class == "actions"
                        || class == "cgu"
                        || class == "tos"
                        || class == "share"
                        || class == "sharing"
                        || class == "partage"
                        || class == "to-top"
                        || class == "viral"
                        // https://www.lexpress.fr/societe/cote-de-popularite-fillon-chute-hamon-grimpe_1879405.html
                        // || class == "header"
                        || class == "bubble"

                        || class == "syndication"
                        || class == "embedded"
                        || class == "embed"
                        || class == "newsletter"
                        || class == "cookie"
                        || class == "cookies"
                        || class == "tags"
                        || class == "tags"
                        || class == "tag-list"
                        || class.starts_with("breadcrumb")
                        || class == "popover"
                        || class == "modal"
                        || class == "sticky"
                        || class == "fixed"
                        || class == "modals"
                        || class == "carousel"
                        || class.starts_with("affiliat")
                        // || class == "meta"
                        || class == "widget"
                        || class.ends_with("author")
                        || class == "hidden"
                        || class == "hide"
                        || class.starts_with("byline")
                        || class.contains("copyright")
                        || class.contains("download")
                        || class.starts_with("rating")
                        || class.starts_with("widget")
                        || class.starts_with("attachment")
                        || class.starts_with("timestamp")
                        || class == "user-info"
                        || class == "user-profile"
                        || class == "credits"
                        || class == "noskim"
                        || class == "logo"
                        || class == "ad"
                        || class == "advertising"
                        || class == "next"
                        || class == "stories"
                        || class == "most-popular"
                        || class == "mol-factobox"
                        || class == "ZendeskForm"
                        || class == "message-container"
                        || class == "yin"
                        || class == "zlylin"
                        || class == "xg1"
                        || class == "bmdh"
                        || class == "viewport"
                        || class == "slide"
                        || class == "overlay"
                        || class == "paid"
                        || class == "toast"
                        || class == "paidcontent"
                        || class == "notloaded"
                        || class == "obsfuscated"
                        || class == "blurred"
                        || class == "nocomments"
                        || class == "reply"
                        || class == "akismet"
                        || class == "noprint"
                        || class == "comments"
                        || class == "commentaire"
                ) || (
                    class.iter().flat_map(|class| class.split_whitespace().map(str::to_owned)))
                    .any(|class|
                        class == "no-display" || class == "no-show"
                            || class.starts_with("footer")
                            || class == "mdl-layout__header"
                            || class == "not-loaded"
                            || class.starts_with("paginat")
                            || class.starts_with("bread-crumb")
                            || class.starts_with("skip")
                            || class.starts_with("md-skip")
                            || class.starts_with("foot")

                            // Negative sample: https://pressroom.natixis.com/actualites/herve-loarer-nomme-responsable-de-la-direction-technologies-infogerance-des-systemes-dinformation-et-services-partages-6683-b09b6.html
                            || class.starts_with("social")
                            || class.starts_with("filters")
                            || class == "dialog"

                            || class == "indicateur-langue"
                            || class.starts_with("wm-ipp")
                            || class.starts_with("mw-gallery")
                            || class == "mw-editsection"
                            || class == "mw-references-wrap"
                            || class == "mw-references-columns"
                            || class == "references-small"
                            || class == "reference"
                            || class == "mw-jump-link"
                            || class.starts_with("cite_ref-")
                            || class == "hatnote"
                            || class == "need_ref"
                            || class == "need_ref_tag"
                            || class.starts_with("bandeau-niveau-")
                            || class.starts_with("bandeau-niveau-")
                            || class.starts_with("bandeau-niveau-")
                            || class == "bandeau-portail"
                            || class == "printfooter"
                            || class == "metadata"
                    ));

            match attrs_
                .iter()
                .filter(|attr| &attr.name.local == "style")
                .map(|attr| attr.value.as_ref())
                .next() {
                Some(style) => {
                    let style = style.to_lowercase();
                    if style.contains("display: none") || style.contains("display:none") || style.contains("position:fixed") {
                        to_skip = true;
                    }
                }
                None => {}
            }

            // Check aria-hidden

            if to_skip {
                return Ok(Nothing);
            }

            let RenderInput {
                ref handle,
                ref parent_style,
            } = input;

            #[cfg(feature = "css")]
            let computed = {
                let computed =
                    context
                        .style_data
                        .computed_style(**parent_style, handle, context.use_doc_css);
                if let Some(true) = computed.display_none.val() {
                    return Ok(Nothing);
                }
                computed
            };
            #[cfg(not(feature = "css"))]
            let computed = **parent_style;

            let result = match name.expanded() {
                expanded_name!(html "html") | expanded_name!(html "body") => {
                    /* process children, but don't add anything */
                    pending(input, move |_, cs| {
                        if !check_container(&cs) {
                            return None;
                        }
                        Some(RenderNode::new_styled(Container(cs), computed))
                    })
                }
                expanded_name!(html "link")
                | expanded_name!(html "meta")
                | expanded_name!(html "script")
                | expanded_name!(html "style")
                | expanded_name!(html "head")
                | expanded_name!(html "button")
                // | expanded_name!(html "header") but some sites use it as a nav bar
                | expanded_name!(html "nav")
                | expanded_name!(html "aside")
                | expanded_name!(html "iframe")
                | expanded_name!(html "figure")  // for wi
                | expanded_name!(html "noscript")
                | expanded_name!(html "footer") => {
                    /* Ignore the head and its children */
                    Nothing
                }
                expanded_name!(html "span") => {
                    /* process children, but don't add anything */
                    pending_noempty(input, move |_, cs| {
                        if !check_container(&cs) {
                            return None;
                        }
                        Some(RenderNode::new_styled(Container(cs), computed))
                    })
                }
                expanded_name!(html "form") => {
                    pending(input, move |_, cs| {
                        // if !check_container(&cs) {
                        //     return None;
                        // }
                        // Some(RenderNode::new_styled(Container(cs), computed))
                        Some(RenderNode::new(Form))
                    })
                }
                expanded_name!(html "a") => {
                    let borrowed = attrs.borrow();
                    let mut target = None;
                    frag_from_name_attr = true;
                    for attr in borrowed.iter() {
                        if &attr.name.local == "href" {
                            target = Some(&*attr.value);
                            break;
                        }
                    }
                    PendingChildren {
                        children: input.children(),
                        cons: if let Some(href) = target {
                            let href: String = href.into();
                            Box::new(move |_, cs: Vec<RenderNode>| {
                                if cs.iter().any(|c| !c.is_shallow_empty()) {
                                    Ok(Some(RenderNode::new_styled(Link(href, cs), computed)))
                                } else {
                                    Ok(None)
                                }
                            })
                        } else {
                            Box::new(move |_, cs| {
                                if !check_container(&cs) {
                                    return Ok(None);
                                }
                                Ok(Some(RenderNode::new_styled(Container(cs), computed)))
                            })
                        },
                        prefn: None,
                        postfn: None,
                    }
                }
                expanded_name!(html "em")
                | expanded_name!(html "i")
                | expanded_name!(html "ins") => pending(input, move |_, cs| {
                    Some(RenderNode::new_styled(Em(cs), computed))
                }),
                expanded_name!(html "strong") => pending(input, move |_, cs| {
                    Some(RenderNode::new_styled(Strong(cs), computed))
                }),
                expanded_name!(html "s") | expanded_name!(html "del") => {
                    pending(input, move |_, cs| {
                        Some(RenderNode::new_styled(Strikeout(cs), computed))
                    })
                }
                expanded_name!(html "code") => pending(input, move |_, cs| {
                    Some(RenderNode::new_styled(Code(cs), computed))
                }),
                expanded_name!(html "img") => {
                    let borrowed = attrs.borrow();
                    let mut title = None;
                    let mut src = None;
                    for attr in borrowed.iter() {
                        if &attr.name.local == "alt" && !attr.value.is_empty() {
                            title = Some(&*attr.value);
                            // If title starts with "{\displaystyle " and ends with "}", remove them
                            if title.unwrap().starts_with("{\\displaystyle ") && title.unwrap().ends_with("}") {
                                title = Some(&title.unwrap()[14..title.unwrap().len() - 1]);
                            }
                        }
                        if &attr.name.local == "src" && !attr.value.is_empty() {
                            src = Some(&*attr.value);
                        }
                        if title.is_some() && src.is_some() {
                            break;
                        }
                    }
                    if let (Some(title), Some(src)) = (title, src) {
                        Finished(RenderNode::new_styled(
                            Img(src.into(), title.into()),
                            computed,
                        ))
                    } else {
                        Nothing
                    }
                }
                expanded_name!(html "h1")
                | expanded_name!(html "h2")
                | expanded_name!(html "h3")
                | expanded_name!(html "h4") => {
                    let level: usize = name.local[1..].parse().unwrap();
                    let header_text = input
                        .handle
                        .children
                        .borrow()
                        .iter()
                        .filter_map(|child| {
                            // println!("child: {:?}", child);
                            if let markup5ever_rcdom::NodeData::Text { contents } = &child.data {
                                Some(contents.borrow().to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    // for wikipedia
                    // println!("header_text: {:?}", header_text);
                    if level == 2 {
                        match (header_text.trim()) {
                            "Notes et références" | "Voir aussi" | "Annexes" | "Compléments" | "Références" | "Référence" | "Liens externes" | "Liens internes" | "Articles connexes" => {
                                // println!("Skip from header_text: {:?}", header_text);
                                context.done = true;
                            }
                            _ => {}
                        }
                    }
                    pending(input, move |_, cs| {
                        Some(RenderNode::new_styled(Header(level, cs), computed))
                    })
                }
                expanded_name!(html "p")
                | expanded_name!(html "article") => pending_noempty(input, move |_, cs| {
                    Some(RenderNode::new_styled(Block(cs), computed))
                }),
                expanded_name!(html "li") => pending(input, move |_, cs| {
                    Some(RenderNode::new_styled(ListItem(cs), computed))
                }),
                expanded_name!(html "sup") => pending(input, move |_, cs| {
                    Some(RenderNode::new_styled(Sup(cs), computed))
                }),
                expanded_name!(html "div")
                | expanded_name!(html "caption") => pending_noempty(input, move |_, cs| {
                    if !check_container(&cs) {
                        return None;
                    }
                    Some(RenderNode::new_styled(Div(cs), computed))
                }),
                expanded_name!(html "pre") => pending(input, move |_, cs| {
                    let mut computed = computed;
                    computed.white_space.maybe_update(
                        false,
                        StyleOrigin::Agent,
                        Default::default(),
                        WhiteSpace::Pre,
                    );
                    computed.internal_pre = true;
                    Some(RenderNode::new_styled(Block(cs), computed))
                }),
                expanded_name!(html "br") => Finished(RenderNode::new_styled(Break, computed)),
                expanded_name!(html "hr") => Finished(RenderNode::new_styled(Break, computed)),
                expanded_name!(html "table") => table_to_render_tree(input, computed, err_out),
                expanded_name!(html "thead") | expanded_name!(html "tbody") => {
                    tbody_to_render_tree(input, computed, err_out)
                }
                expanded_name!(html "tr") => tr_to_render_tree(input, computed, err_out),
                expanded_name!(html "th") | expanded_name!(html "td") => {
                    td_to_render_tree(input, computed, err_out)
                }
                expanded_name!(html "blockquote") => pending_noempty(input, move |_, cs| {
                    Some(RenderNode::new_styled(BlockQuote(cs), computed))
                }),
                expanded_name!(html "ul") => pending_noempty(input, move |_, cs| {
                    Some(RenderNode::new_styled(Ul(cs), computed))
                }),
                expanded_name!(html "ol") => {
                    let borrowed = attrs.borrow();
                    let mut start = 1;
                    for attr in borrowed.iter() {
                        if &attr.name.local == "start" {
                            start = attr.value.parse().ok().unwrap_or(1);
                            break;
                        }
                    }

                    pending_noempty(input, move |_, cs| {
                        // There can be extra nodes which aren't ListItem (like whitespace text
                        // nodes).  We need to filter those out to avoid messing up the rendering.
                        let cs = cs
                            .into_iter()
                            .filter(|n| matches!(n.info, RenderNodeInfo::ListItem(..)))
                            .collect();
                        Some(RenderNode::new_styled(Ol(start, cs), computed))
                    })
                }
                expanded_name!(html "dl") => Finished(RenderNode::new_styled(
                    Dl(desc_list_children_to_render_nodes(
                        handle.clone(),
                        err_out,
                        context,
                    )?),
                    computed,
                )),
                _ => {
                    html_trace!("Unhandled element: {:?}\n", name.local);
                    pending_noempty(input, move |_, cs| {
                        if !check_container(&cs) {
                            return None;
                        }
                        Some(RenderNode::new_styled(Container(cs), computed))
                    })
                }
            };

            let mut fragment = None;
            let borrowed = attrs.borrow();
            for attr in borrowed.iter() {
                if &attr.name.local == "id" || (frag_from_name_attr && &attr.name.local == "name") {
                    fragment = Some(attr.value.to_string());
                    break;
                }
            }

            let Some(fragname) = fragment else {
                return Ok(result);
            };
            match result {
                Finished(node) => {
                    Finished(prepend_marker(RenderNode::new(FragStart(fragname)), node))
                }
                Nothing => Finished(RenderNode::new(FragStart(fragname))),
                PendingChildren {
                    children,
                    cons,
                    prefn,
                    postfn,
                } => PendingChildren {
                    children,
                    prefn,
                    postfn,
                    cons: Box::new(move |ctx, ch| {
                        let fragnode = RenderNode::new(FragStart(fragname));
                        match cons(ctx, ch)? {
                            None => Ok(Some(fragnode)),
                            Some(node) => Ok(Some(prepend_marker(fragnode, node))),
                        }
                    }),
                },
            }
        }
        markup5ever_rcdom::NodeData::Text { contents: ref tstr } => {
            // Check template_content for forbidden text: "À lire aussi", "S'abonner", "Abonnez vous", "Abonne toi":
            match String::from(&*tstr.borrow()).to_lowercase().as_str() {
                "à lire aussi" | "a lire aussi" | "s'abonner" | "abonnez vous" | "abonne toi" | "en accès libre" | "en accès libre" => {
                    return Ok(Nothing);
                }
                _ => {}
            }
            Finished(RenderNode::new(Text((&*tstr.borrow()).into())))
        }
        _ => {
            // NodeData doesn't have a Debug impl.
            writeln!(err_out, "Unhandled node type.").unwrap();
            Nothing
        }
    })
}

fn render_tree_to_string<T: Write, D: TextDecorator>(
    context: &mut HtmlContext,
    renderer: SubRenderer<D>,
    decorator: &D,
    tree: RenderNode,
    err_out: &mut T,
) -> Result<SubRenderer<D>> {
    /* Phase 1: get size estimates. */
    // can't actually error, but Ok-wrap to satisfy tree_map_reduce signature
    tree_map_reduce(context, &tree, |context, node| {
        Ok(precalc_size_estimate(node, context, decorator))
    })?;
    /* Phase 2: actually render. */
    let mut renderer = TextRenderer::new(renderer);
    tree_map_reduce(&mut renderer, tree, |renderer, node| {
        Ok(do_render_node(renderer, node, err_out)?)
    })?;
    let (mut renderer, links) = renderer.into_inner();
    let lines = renderer.finalise(links);
    // And add the links
    if !lines.is_empty() {
        renderer.start_block()?;
        renderer.fmt_links(lines);
    }
    Ok(renderer)
}

fn pending2<
    D: TextDecorator,
    F: FnOnce(
        &mut TextRenderer<D>,
        Vec<Option<SubRenderer<D>>>,
    ) -> Result<Option<Option<SubRenderer<D>>>>
    + 'static,
>(
    children: Vec<RenderNode>,
    f: F,
) -> TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>> {
    TreeMapResult::PendingChildren {
        children,
        cons: Box::new(f),
        prefn: None,
        postfn: None,
    }
}

/// Keep track of what style state has been applied to a renderer so that we
/// can undo it.
#[derive(Default)]
struct PushedStyleInfo {
    colour: bool,
    bgcolour: bool,
    white_space: bool,
    preformat: bool,
}

impl PushedStyleInfo {
    fn apply<D: TextDecorator>(render: &mut TextRenderer<D>, style: &ComputedStyle) -> Self {
        #[allow(unused_mut)]
        let mut result: PushedStyleInfo = Default::default();
        #[cfg(feature = "css")]
        if let Some(col) = style.colour.val() {
            render.push_colour(col);
            result.colour = true;
        }
        #[cfg(feature = "css")]
        if let Some(col) = style.bg_colour.val() {
            render.push_bgcolour(col);
            result.bgcolour = true;
        }
        if let Some(ws) = style.white_space.val() {
            if let WhiteSpace::Pre | WhiteSpace::PreWrap = ws {
                render.push_ws(ws);
                result.white_space = true;
            }
        }
        if style.internal_pre {
            render.push_preformat();
            result.preformat = true;
        }
        result
    }
    fn unwind<D: TextDecorator>(self, renderer: &mut TextRenderer<D>) {
        if self.bgcolour {
            renderer.pop_bgcolour();
        }
        if self.colour {
            renderer.pop_colour();
        }
        if self.white_space {
            renderer.pop_ws();
        }
        if self.preformat {
            renderer.pop_preformat();
        }
    }
}

fn do_render_node<T: Write, D: TextDecorator>(
    renderer: &mut TextRenderer<D>,
    tree: RenderNode,
    err_out: &mut T,
) -> render::Result<TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>>> {
    html_trace!("do_render_node({:?}", tree);
    use RenderNodeInfo::*;
    use TreeMapResult::*;

    let size_estimate = tree.size_estimate.get().unwrap_or_default();

    let pushed_style = PushedStyleInfo::apply(renderer, &tree.style);

    Ok(match tree.info {
        Text(ref tstr) => {
            renderer.add_inline_text(tstr)?;
            pushed_style.unwind(renderer);
            Finished(None)
        }
        Container(children) => pending2(children, |renderer, _| {
            pushed_style.unwind(renderer);
            Ok(Some(None))
        }),
        FragmentContainer(children) => pending2(children, |renderer, _| {
            pushed_style.unwind(renderer);
            Ok(Some(None))
        }),
        Link(href, children) => {
            renderer.start_link(&href)?;
            pending2(children, move |renderer: &mut TextRenderer<D>, _| {
                renderer.end_link()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Em(children) => {
            renderer.start_emphasis()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_emphasis()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Strong(children) => {
            renderer.start_strong()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_strong()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Strikeout(children) => {
            renderer.start_strikeout()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_strikeout()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Code(children) => {
            renderer.start_code()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_code()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Img(src, title) => {
            renderer.add_image(&src, &title)?;
            pushed_style.unwind(renderer);
            Finished(None)
        }
        Block(children) | ListItem(children) => {
            renderer.start_block()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_block();
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Header(level, children) => {
            // debug print lines:
            // println!("-------------");
            // for line in &renderer.lines {
            // //     println!("{:?}", line);
            // }
            renderer.add_empty_line()?;
            let prefix = renderer.header_prefix(level);
            let prefix_size = size_estimate.prefix_size;
            debug_assert!(prefix.len() == prefix_size);
            let min_width = size_estimate.min_width;
            let inner_width = min_width.saturating_sub(prefix_size);
            let sub_builder =
                renderer.new_cased_sub_renderer(renderer.width_minus(prefix_size, inner_width)?, if level <= 2 { Case::Upper } else { Case::Title })?;
            renderer.push(sub_builder);
            pending2(children, move |renderer: &mut TextRenderer<D>, _| {
                let sub_builder = renderer.pop();

                renderer.start_block()?;
                renderer.append_subrender(sub_builder, repeat(&prefix[..]), None)?;
                renderer.end_block();
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Div(children) => {
            renderer.new_line()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.new_line()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        BlockQuote(children) => {
            let prefix = renderer.quote_prefix();
            debug_assert!(size_estimate.prefix_size == prefix.len());
            let inner_width = size_estimate.min_width - prefix.len();
            let sub_builder =
                renderer.new_sub_renderer(renderer.width_minus(prefix.len(), inner_width)?)?;
            renderer.push(sub_builder);
            pending2(children, move |renderer: &mut TextRenderer<D>, _| {
                let sub_builder = renderer.pop();

                renderer.start_block()?;
                renderer.append_subrender(sub_builder, repeat(&prefix[..]), None)?;
                renderer.end_block();
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Ul(items) => {
            renderer.start_block()?;

            let prefix = renderer.unordered_item_prefix();
            let prefix_len = prefix.len();

            TreeMapResult::PendingChildren {
                children: items,
                cons: Box::new(|renderer, _| {
                    pushed_style.unwind(renderer);
                    Ok(Some(None))
                }),
                prefn: Some(Box::new(move |renderer: &mut TextRenderer<D>, _| {
                    let inner_width = size_estimate.min_width - prefix_len;
                    let sub_builder = renderer
                        .new_sub_renderer(renderer.width_minus(prefix_len, inner_width)?)?;
                    renderer.push(sub_builder);
                    Ok(())
                })),
                postfn: Some(Box::new(move |renderer: &mut TextRenderer<D>, _| {
                    let sub_builder = renderer.pop();

                    let indent = " ".repeat(prefix.len());

                    renderer.append_subrender(
                        sub_builder,
                        once(&prefix[..]).chain(repeat(&indent[..])),
                        None,
                    )?;
                    Ok(())
                })),
            }
        }
        Ol(start, items) => {
            renderer.start_block()?;

            let num_items = items.len();

            // The prefix width could be at either end if the start is negative.
            let min_number = start;
            // Assumption: num_items can't overflow isize.
            let max_number = start + (num_items as i64) - 1;
            let prefix_width_min = renderer.ordered_item_prefix(min_number).len();
            let prefix_width_max = 1; //renderer.ordered_item_prefix(max_number).len();
            let prefix_width = max(prefix_width_min, prefix_width_max);
            let prefixn = format!("{: <width$}", "", width = prefix_width);
            let i: Cell<_> = Cell::new(start);

            TreeMapResult::PendingChildren {
                children: items,
                cons: Box::new(|renderer, _| {
                    pushed_style.unwind(renderer);
                    Ok(Some(None))
                }),
                prefn: Some(Box::new(move |renderer: &mut TextRenderer<D>, _| {
                    let inner_min = size_estimate.min_width - size_estimate.prefix_size;
                    let sub_builder = renderer
                        .new_sub_renderer(renderer.width_minus(prefix_width, inner_min)?)?;
                    renderer.push(sub_builder);
                    Ok(())
                })),
                postfn: Some(Box::new(move |renderer: &mut TextRenderer<D>, _| {
                    let sub_builder = renderer.pop();
                    let prefix1 = renderer.ordered_item_prefix(i.get());
                    let prefix1 = format!("{: <width$}", prefix1, width = prefix_width);

                    renderer.append_subrender(
                        sub_builder,
                        once(prefix1.as_str()).chain(repeat(prefixn.as_str())),
                        None,
                    )?;
                    i.set(i.get() + 1);
                    Ok(())
                })),
            }
        }
        Dl(items) => {
            renderer.start_block()?;

            TreeMapResult::PendingChildren {
                children: items,
                cons: Box::new(|renderer, _| {
                    pushed_style.unwind(renderer);
                    Ok(Some(None))
                }),
                prefn: None,
                postfn: None,
            }
        }
        Dt(children) => {
            renderer.new_line()?;
            renderer.start_emphasis()?;
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                renderer.end_emphasis()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Dd(children) => {
            let inner_min = size_estimate.min_width - 2;
            let sub_builder = renderer.new_sub_renderer(renderer.width_minus(2, inner_min)?)?;
            renderer.push(sub_builder);
            pending2(children, |renderer: &mut TextRenderer<D>, _| {
                let sub_builder = renderer.pop();
                renderer.append_subrender(sub_builder, repeat("  "), None)?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Break => {
            renderer.new_line_hard()?;
            pushed_style.unwind(renderer);
            Finished(None)
        }
        Table(tab) => render_table_tree(renderer, tab, err_out)?,
        TableRow(row, false) => render_table_row(renderer, row, pushed_style, err_out),
        TableRow(row, true) => render_table_row_vert(renderer, row, pushed_style, err_out),
        TableBody(_) => unimplemented!("Unexpected TableBody while rendering"),
        TableCell(cell) => render_table_cell(renderer, cell, pushed_style, err_out),
        FragStart(fragname) => {
            renderer.record_frag_start(&fragname);
            pushed_style.unwind(renderer);
            Finished(None)
        }
        Sup(children) => {
            // Special case for digit-only superscripts - use superscript
            // characters.
            let old_case: Case = renderer.case.clone();
            renderer.case = Case::Sup;
            // renderer.start_superscript()?;
            pending2(children, move |renderer: &mut TextRenderer<D>, _| {
                renderer.case = old_case;

                // renderer.end_superscript()?;
                pushed_style.unwind(renderer);
                Ok(Some(None))
            })
        }
        Form => {
            // Don't render the form
            pushed_style.unwind(renderer);
            Finished(None)
        }
    })
}

fn render_table_tree<T: Write, D: TextDecorator>(
    renderer: &mut TextRenderer<D>,
    table: RenderTable,
    _err_out: &mut T,
) -> render::Result<TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>>> {
    /* Now lay out the table. */
    let num_columns = table.num_columns;
    let mut sub_renderer = renderer.new_sub_renderer(renderer.width())?;

    /* Heuristic: scale the column widths according to how much content there is. */
    let mut col_sizes: Vec<SizeEstimate> = vec![Default::default(); num_columns];

    for row in table.rows() {
        let mut colno = 0;
        for cell in row.cells() {
            // FIXME: get_size_estimate is still recursive.
            let mut estimate = cell.get_size_estimate();

            // If the cell has a colspan>1, then spread its size between the
            // columns.
            // estimate.size /= cell.colspan;
            // estimate.min_width /= cell.colspan;
            for i in 0..cell.colspan {
                // let border = if (colnum + colno > 0) { 1 } else { 0 };
                col_sizes[colno + i].size = max(
                    col_sizes[colno + i].size,
                    estimate.size / cell.colspan, // + border,
                );
                col_sizes[colno + i].min_width = max(
                    col_sizes[colno + i].min_width,
                    estimate.min_width / cell.colspan, // + border,
                );
            }
            colno += cell.colspan;
        }
    }
    // TODO: remove empty columns
    let tot_size: usize = col_sizes.iter().map(|est| est.size).sum();
    let min_size: usize = col_sizes.iter().map(|est| est.min_width).sum::<usize>()
        + col_sizes.len().saturating_sub(1);
    let width = sub_renderer.width();

    let vert_row = sub_renderer.options.raw || (min_size > width || width == 0);

    let mut col_widths: Vec<usize> = if !vert_row {
        col_sizes
            .iter()
            .map(|sz| {
                if sz.size == 0 {
                    0
                } else {
                    min(
                        sz.size,
                        if usize::MAX / width <= sz.size {
                            // The provided width is too large to multiply by width,
                            // so do it the other way around.
                            max((width / tot_size) * sz.size, sz.min_width)
                        } else {
                            max(sz.size * width / tot_size, sz.min_width)
                        },
                    )
                }
            })
            .collect()
    } else {
        col_sizes.iter().map(|_| width).collect()
    };

    if !vert_row {
        let num_cols = col_widths.len();
        if num_cols > 0 {
            loop {
                let cur_width = col_widths.iter().sum::<usize>(); // + num_cols - 1;
                if cur_width <= width {
                    break;
                }
                let (i, _) = col_widths
                    .iter()
                    .enumerate()
                    .max_by_key(|&(colno, width)| {
                        (
                            width.saturating_sub(col_sizes[colno].min_width),
                            width,
                            usize::MAX - colno,
                        )
                    })
                    .unwrap();
                col_widths[i] -= 1;
            }
        }
    }

    let table_width = if vert_row {
        width
    } else {
        col_widths.iter().cloned().sum::<usize>()
            + col_widths
            .iter()
            .filter(|&w| w > &0)
            .count()
            .saturating_sub(1)
    };

    if table_width != 0 && sub_renderer.options.draw_borders {
        // sub_renderer.add_horizontal_border_width(table_width)?;
    }

    renderer.push(sub_renderer);
    Ok(TreeMapResult::PendingChildren {
        children: table.into_rows(col_widths, vert_row),
        cons: Box::new(move |renderer: &mut TextRenderer<D>, _| {
            let mut sub_renderer = renderer.pop();
            sub_renderer.new_line_hard()?;
            // iterate over sub renderer chars
            renderer.append_subrender(
                sub_renderer,
                repeat(""),
                Some(|strings: &Vec<String>| {
                    let mut total_digits = 1;
                    let mut total_non_punct = 1;
                    strings.iter().map(|s| s.chars())
                        .flatten()
                        .for_each(|c| match c {
                            '0'..='9' => total_digits += 1,
                            'a'..='z' | 'A'..='Z' => total_non_punct += 1,
                            _ => (),
                        });
                    // println!("Total digits: {} / Total non punct {}", total_digits, total_non_punct);
                    (total_non_punct as f64 / total_digits as f64) > 4.0
                }),
            )?;
            Ok(Some(None))
        }),
        prefn: None,
        postfn: None,
    })
}

fn render_table_row<T: Write, D: TextDecorator>(
    _renderer: &mut TextRenderer<D>,
    row: RenderTableRow,
    pushed_style: PushedStyleInfo,
    _err_out: &mut T,
) -> TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>> {
    TreeMapResult::PendingChildren {
        children: row.into_cells(false),
        cons: Box::new(|builders, children| {
            let children: Vec<_> = children.into_iter().map(Option::unwrap).collect();
            if children.iter().any(|c| !c.empty()) {
                builders.append_columns_with_borders(children, true)?;
            }
            pushed_style.unwind(builders);
            Ok(Some(None))
        }),
        prefn: Some(Box::new(|renderer: &mut TextRenderer<D>, node| {
            if let RenderNodeInfo::TableCell(ref cell) = node.info {
                let sub_builder = renderer.new_sub_renderer(cell.col_width.unwrap())?;
                renderer.push(sub_builder);
                Ok(())
            } else {
                panic!()
            }
        })),
        postfn: Some(Box::new(|_renderer: &mut TextRenderer<D>, _| Ok(()))),
    }
}

fn render_table_row_vert<T: Write, D: TextDecorator>(
    _renderer: &mut TextRenderer<D>,
    row: RenderTableRow,
    pushed_style: PushedStyleInfo,
    _err_out: &mut T,
) -> TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>> {
    TreeMapResult::PendingChildren {
        children: row.into_cells(true),
        cons: Box::new(|builders, children| {
            let children: Vec<_> = children.into_iter().map(Option::unwrap).collect();
            builders.append_vert_row(children)?;
            pushed_style.unwind(builders);
            Ok(Some(None))
        }),
        prefn: Some(Box::new(|renderer: &mut TextRenderer<D>, node| {
            if let RenderNodeInfo::TableCell(ref cell) = node.info {
                let sub_builder = renderer.new_sub_renderer(cell.col_width.unwrap())?;
                renderer.push(sub_builder);
                Ok(())
            } else {
                Err(Error::Fail)
            }
        })),
        postfn: Some(Box::new(|_renderer: &mut TextRenderer<D>, _| Ok(()))),
    }
}

fn render_table_cell<T: Write, D: TextDecorator>(
    _renderer: &mut TextRenderer<D>,
    cell: RenderTableCell,
    pushed_style: PushedStyleInfo,
    _err_out: &mut T,
) -> TreeMapResult<'static, TextRenderer<D>, RenderNode, Option<SubRenderer<D>>> {
    pending2(cell.content, |renderer: &mut TextRenderer<D>, _| {
        pushed_style.unwind(renderer);
        let sub_builder = renderer.pop();

        Ok(Some(Some(sub_builder)))
    })
}

pub mod config {
    //! Configure the HTML to text translation using the `Config` type, which can be
    //! constructed using one of the functions in this module.
    use std::io;

    use super::Error;
    #[cfg(feature = "css")]
    use crate::css::StyleData;
    use crate::{
        render::text_renderer::{
            PlainDecorator, RichAnnotation, RichDecorator, TaggedLine, TextDecorator,
        },
        HtmlContext, RenderTree, Result, MIN_WIDTH,
    };

    /// Configure the HTML processing.
    pub struct Config<D: TextDecorator> {
        decorator: D,

        max_wrap_width: Option<usize>,

        #[cfg(feature = "css")]
        style: StyleData,
        #[cfg(feature = "css")]
        use_doc_css: bool,

        pad_block_width: bool,

        allow_width_overflow: bool,
        min_wrap_width: usize,
        raw: bool,
        draw_borders: bool,
        wrap_links: bool,
    }

    impl<D: TextDecorator> Config<D> {
        /// Make the HtmlContext from self.
        fn make_context(&self) -> HtmlContext {
            HtmlContext {
                #[cfg(feature = "css")]
                style_data: self.style.clone(),
                #[cfg(feature = "css")]
                use_doc_css: self.use_doc_css,

                max_wrap_width: self.max_wrap_width,
                pad_block_width: self.pad_block_width,
                allow_width_overflow: self.allow_width_overflow,
                min_wrap_width: self.min_wrap_width,
                raw: self.raw,
                draw_borders: self.draw_borders,
                wrap_links: self.wrap_links,
                done: false,
            }
        }
        /// Parse with context.
        fn do_parse<R: io::Read>(
            &mut self,
            context: &mut HtmlContext,
            input: R,
        ) -> Result<RenderTree> {
            super::parse_with_context(input, context)
        }

        /// Parse the HTML into a DOM structure.
        pub fn parse_html<R: io::Read>(&self, mut input: R) -> Result<super::RcDom> {
            use html5ever::tendril::TendrilSink;
            let opts = super::ParseOpts {
                tree_builder: super::TreeBuilderOpts {
                    drop_doctype: true,
                    ..Default::default()
                },
                ..Default::default()
            };
            Ok(super::parse_document(super::RcDom::default(), opts)
                .from_utf8()
                .read_from(&mut input)?)
        }

        /// Convert an HTML DOM into a RenderTree.
        pub fn dom_to_render_tree(&self, dom: &super::RcDom) -> Result<RenderTree> {
            Ok(RenderTree(
                super::dom_to_render_tree_with_context(
                    dom.document.clone(),
                    &mut io::sink(),
                    &mut self.make_context(),
                )?
                    .ok_or(Error::Fail)?,
            ))
        }

        /// Render an existing RenderTree into a string.
        pub fn render_to_string(&self, render_tree: RenderTree, width: usize) -> Result<String> {
            let s = render_tree
                .render_with_context(
                    &mut self.make_context(),
                    width,
                    self.decorator.make_subblock_decorator(),
                )?
                .into_string()?;
            Ok(s)
        }

        /// Take an existing RenderTree, and returns text wrapped to `width` columns.
        /// The text is returned as a `Vec<TaggedLine<_>>`; the annotations are vectors
        /// of the provided text decorator's `Annotation`.  The "outer" annotation comes first in
        /// the `Vec`.
        pub fn render_to_lines(
            &self,
            render_tree: RenderTree,
            width: usize,
        ) -> Result<Vec<TaggedLine<Vec<D::Annotation>>>> {
            render_tree
                .render_with_context(
                    &mut self.make_context(),
                    width,
                    self.decorator.make_subblock_decorator(),
                )?
                .into_lines()
        }

        /// Reads HTML from `input`, and returns a `String` with text wrapped to
        /// `width` columns.
        pub fn string_from_read<R: std::io::Read>(
            mut self,
            input: R,
            width: usize,
        ) -> Result<String> {
            let mut context = self.make_context();
            let s = self
                .do_parse(&mut context, input)?
                .render_with_context(&mut context, width, self.decorator)?
                .into_string()?;
            Ok(s)
        }

        /// Reads HTML from `input`, and returns text wrapped to `width` columns.
        /// The text is returned as a `Vec<TaggedLine<_>>`; the annotations are vectors
        /// of the provided text decorator's `Annotation`.  The "outer" annotation comes first in
        /// the `Vec`.
        pub fn lines_from_read<R: std::io::Read>(
            mut self,
            input: R,
            width: usize,
        ) -> Result<Vec<TaggedLine<Vec<D::Annotation>>>> {
            let mut context = self.make_context();
            self.do_parse(&mut context, input)?
                .render_with_context(&mut context, width, self.decorator)?
                .into_lines()
        }

        #[cfg(feature = "css")]
        /// Add some CSS rules which will be used (if supported) with any
        /// HTML processed.
        pub fn add_css(mut self, css: &str) -> Result<Self> {
            self.style.add_user_css(css)?;
            Ok(self)
        }

        #[cfg(feature = "css")]
        /// Add some agent CSS rules which will be used (if supported) with any
        /// HTML processed.
        pub fn add_agent_css(mut self, css: &str) -> Result<Self> {
            self.style.add_agent_css(css)?;
            Ok(self)
        }

        #[cfg(feature = "css")]
        /// Parse CSS from any \<style\> elements and use supported rules.
        pub fn use_doc_css(mut self) -> Self {
            self.use_doc_css = true;
            self
        }

        /// Pad lines out to the full render width.
        pub fn pad_block_width(mut self) -> Self {
            self.pad_block_width = true;
            self
        }

        /// Set the maximum text wrap width.
        /// When set, paragraphs will be wrapped to that width even if there
        /// is more total width available for rendering.
        pub fn max_wrap_width(mut self, wrap_width: usize) -> Self {
            self.max_wrap_width = Some(wrap_width);
            self
        }

        /// Allow the output to be wider than the max width.  When enabled,
        /// then output wider than the specified width will be returned
        /// instead of returning `Err(TooNarrow)` if the output wouldn't
        /// otherwise fit.
        pub fn allow_width_overflow(mut self) -> Self {
            self.allow_width_overflow = true;
            self
        }

        /// Set the minimum width for text wrapping.  The default is 3.
        /// Blocks of text will be forced to have at least this width
        /// (unless the text inside is less than that).  Increasing this
        /// can increase the chance that the width will overflow, leading
        /// to a TooNarrow error unless `allow_width_overflow()` is set.
        pub fn min_wrap_width(mut self, min_wrap_width: usize) -> Self {
            self.min_wrap_width = min_wrap_width;
            self
        }

        /// Raw extraction, ensures text in table cells ends up rendered together
        /// This traverses tables as if they had a single column and every cell is its own row.
        /// Implies `no_table_borders()`
        pub fn raw_mode(mut self, raw: bool) -> Self {
            self.raw = raw;
            self.draw_borders = false;
            self
        }

        /// Do not render table borders
        pub fn no_table_borders(mut self) -> Self {
            self.draw_borders = false;
            self
        }
        /// Do not wrap links
        pub fn no_link_wrapping(mut self) -> Self {
            self.wrap_links = false;
            self
        }
    }

    impl Config<RichDecorator> {
        /// Return coloured text.  `colour_map` is a function which takes
        /// a list of `RichAnnotation` and some text, and returns the text
        /// with any terminal escapes desired to indicate those annotations
        /// (such as colour).
        pub fn coloured<R, FMap>(
            mut self,
            input: R,
            width: usize,
            colour_map: FMap,
        ) -> Result<String>
        where
            R: std::io::Read,
            FMap: Fn(&[RichAnnotation], &str) -> String,
        {
            let mut context = self.make_context();
            let lines = self
                .do_parse(&mut context, input)?
                .render_with_context(&mut context, width, self.decorator)?
                .into_lines()?;

            let mut result = String::new();
            for line in lines {
                for ts in line.tagged_strings() {
                    result.push_str(&colour_map(&ts.tag, &ts.s));
                }
                result.push('\n');
            }
            Ok(result)
        }

        /// Return coloured text from a RenderTree.  `colour_map` is a function which takes a list
        /// of `RichAnnotation` and some text, and returns the text with any terminal escapes
        /// desired to indicate those annotations (such as colour).
        pub fn render_coloured<FMap>(
            &self,
            render_tree: RenderTree,
            width: usize,
            colour_map: FMap,
        ) -> Result<String>
        where
            FMap: Fn(&[RichAnnotation], &str) -> String,
        {
            let lines = self.render_to_lines(render_tree, width)?;

            let mut result = String::new();
            for line in lines {
                for ts in line.tagged_strings() {
                    result.push_str(&colour_map(&ts.tag, &ts.s));
                }
                result.push('\n');
            }
            Ok(result)
        }
    }

    /// Return a Config initialized with a `RichDecorator`.
    pub fn rich() -> Config<RichDecorator> {
        Config {
            decorator: RichDecorator::new(),
            #[cfg(feature = "css")]
            style: Default::default(),
            #[cfg(feature = "css")]
            use_doc_css: false,
            max_wrap_width: None,
            pad_block_width: false,
            allow_width_overflow: false,
            min_wrap_width: MIN_WIDTH,
            raw: false,
            draw_borders: true,
            wrap_links: true,
        }
    }

    /// Return a Config initialized with a `PlainDecorator`.
    pub fn plain() -> Config<PlainDecorator> {
        Config {
            decorator: PlainDecorator::new(),
            #[cfg(feature = "css")]
            style: Default::default(),
            #[cfg(feature = "css")]
            use_doc_css: false,
            max_wrap_width: None,
            pad_block_width: false,
            allow_width_overflow: false,
            min_wrap_width: MIN_WIDTH,
            raw: false,
            draw_borders: true,
            wrap_links: true,
        }
    }

    /// Return a Config initialized with a custom decorator.
    pub fn with_decorator<D: TextDecorator>(decorator: D) -> Config<D> {
        Config {
            decorator,
            #[cfg(feature = "css")]
            style: Default::default(),
            #[cfg(feature = "css")]
            use_doc_css: false,
            max_wrap_width: None,
            pad_block_width: false,
            allow_width_overflow: false,
            min_wrap_width: MIN_WIDTH,
            raw: false,
            draw_borders: true,
            wrap_links: true,
        }
    }
}

/// The structure of an HTML document that can be rendered using a [`TextDecorator`][].
///
/// [`TextDecorator`]: render/text_renderer/trait.TextDecorator.html

#[derive(Clone, Debug)]
pub struct RenderTree(RenderNode);

impl std::fmt::Display for RenderTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Render tree:")?;
        self.0.write_self(f, 1)
    }
}

impl RenderTree {
    /// Render this document using the given `decorator` and wrap it to `width` columns.
    fn render_with_context<D: TextDecorator>(
        self,
        context: &mut HtmlContext,
        width: usize,
        decorator: D,
    ) -> Result<RenderedText<D>> {
        if width == 0 {
            return Err(Error::TooNarrow);
        }
        let render_options = RenderOptions {
            wrap_width: context.max_wrap_width,
            pad_block_width: context.pad_block_width,
            allow_width_overflow: context.allow_width_overflow,
            raw: context.raw,
            draw_borders: context.draw_borders,
            wrap_links: context.wrap_links,
        };
        let test_decorator = decorator.make_subblock_decorator();
        let builder = SubRenderer::new(width, render_options, decorator);
        let builder =
            render_tree_to_string(context, builder, &test_decorator, self.0, &mut io::sink())?;
        Ok(RenderedText(builder))
    }

    /// Render this document using the given `decorator` and wrap it to `width` columns.
    fn render<D: TextDecorator>(self, width: usize, decorator: D) -> Result<RenderedText<D>> {
        self.render_with_context(&mut Default::default(), width, decorator)
    }
}

/// A rendered HTML document.
struct RenderedText<D: TextDecorator>(SubRenderer<D>);

impl<D: TextDecorator> RenderedText<D> {
    /// Convert the rendered HTML document to a string.
    fn into_string(self) -> render::Result<String> {
        self.0.into_string()
    }

    /// Convert the rendered HTML document to a vector of lines with the annotations created by the
    /// decorator.
    fn into_lines(self) -> Result<Vec<TaggedLine<Vec<D::Annotation>>>> {
        Ok(self
            .0
            .into_lines()?
            .into_iter()
            .map(RenderLine::into_tagged_line)
            .collect())
    }
}

fn parse_with_context(mut input: impl io::Read, context: &mut HtmlContext) -> Result<RenderTree> {
    let opts = ParseOpts {
        tree_builder: TreeBuilderOpts {
            drop_doctype: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let dom = parse_document(RcDom::default(), opts)
        .from_utf8()
        .read_from(&mut input)?;
    let render_tree =
        dom_to_render_tree_with_context(dom.document.clone(), &mut io::sink(), context)?
            .ok_or(Error::Fail)?;
    Ok(RenderTree(render_tree))
}

/// Reads and parses HTML from `input` and prepares a render tree.
pub fn parse(input: impl io::Read) -> Result<RenderTree> {
    parse_with_context(input, &mut Default::default())
}

/// Reads HTML from `input`, decorates it using `decorator`, and
/// returns a `String` with text wrapped to `width` columns.
pub fn from_read_with_decorator<R, D>(input: R, width: usize, decorator: D) -> Result<String>
where
    R: io::Read,
    D: TextDecorator,
{
    config::with_decorator(decorator).string_from_read(input, width)
}

/// Reads HTML from `input`, and returns a `String` with text wrapped to
/// `width` columns.
pub fn from_read<R>(input: R, width: usize) -> Result<String>
where
    R: io::Read,
{
    config::plain().string_from_read(input, width)
}

/// Reads HTML from `input`, and returns text wrapped to `width` columns.
/// The text is returned as a `Vec<TaggedLine<_>>`; the annotations are vectors
/// of `RichAnnotation`.  The "outer" annotation comes first in the `Vec`.
pub fn from_read_rich<R>(input: R, width: usize) -> Result<Vec<TaggedLine<Vec<RichAnnotation>>>>
where
    R: io::Read,
{
    config::rich().lines_from_read(input, width)
}

mod ansi_colours;

pub use ansi_colours::from_read_coloured;

#[cfg(test)]
mod tests;

fn calc_ol_prefix_size<D: TextDecorator>(start: i64, num_items: usize, decorator: &D) -> usize {
    // The prefix width could be at either end if the start is negative.
    let min_number = start;
    // Assumption: num_items can't overflow isize.
    let max_number = start + (num_items as i64) - 1;

    // This assumes that the decorator gives the same width as default.
    let prefix_width_min = decorator.ordered_item_prefix(min_number).len();
    let prefix_width_max = decorator.ordered_item_prefix(max_number).len();
    max(prefix_width_min, prefix_width_max)
}
