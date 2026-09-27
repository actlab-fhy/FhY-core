//! A serde serializer that builds the Python objects `json.loads` would make
//! of a value's JSON text, without writing the text.
//!
//! Maps and structs become `dict`s in the order the value writes them,
//! sequences and tuples `list`s, strings `str`, integers `int`, floats
//! `float`, and `None` and unit values `None`; an enum variant is the
//! externally tagged form `serde_json` writes, `"variant"` or `{"variant":
//! content}`. So `serialize_to_dict()` equals `json.loads(to_json())`.

use std::fmt;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString};
use serde::ser::{self, Serialize};

/// The error of a value serde cannot write.
#[derive(Debug)]
pub(super) struct Error(pub(super) String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

impl From<PyErr> for Error {
    fn from(error: PyErr) -> Self {
        Self(error.to_string())
    }
}

/// Return the Python object of `value`.
pub(super) fn to_python<'py, T: Serialize + ?Sized>(
    py: Python<'py>,
    value: &T,
) -> Result<Bound<'py, PyAny>, Error> {
    value.serialize(Serializer { py })
}

/// The serializer into Python objects.
#[derive(Clone, Copy)]
struct Serializer<'py> {
    py: Python<'py>,
}

/// Return `{variant: content}`.
fn tagged<'py>(
    py: Python<'py>,
    variant: &'static str,
    content: Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, Error> {
    let dict = PyDict::new(py);
    dict.set_item(variant, content)?;
    Ok(dict.into_any())
}

impl<'py> ser::Serializer for Serializer<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;
    type SerializeSeq = Seq<'py>;
    type SerializeTuple = Seq<'py>;
    type SerializeTupleStruct = Seq<'py>;
    type SerializeTupleVariant = Seq<'py>;
    type SerializeMap = Map<'py>;
    type SerializeStruct = Map<'py>;
    type SerializeStructVariant = Map<'py>;

    fn serialize_bool(self, value: bool) -> Result<Self::Ok, Error> {
        Ok(PyBool::new(self.py, value).to_owned().into_any())
    }

    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Error> {
        self.serialize_i64(value.into())
    }

    fn serialize_i16(self, value: i16) -> Result<Self::Ok, Error> {
        self.serialize_i64(value.into())
    }

    fn serialize_i32(self, value: i32) -> Result<Self::Ok, Error> {
        self.serialize_i64(value.into())
    }

    fn serialize_i64(self, value: i64) -> Result<Self::Ok, Error> {
        let Ok(object) = value.into_pyobject(self.py);
        Ok(object.into_any())
    }

    fn serialize_u8(self, value: u8) -> Result<Self::Ok, Error> {
        self.serialize_u64(value.into())
    }

    fn serialize_u16(self, value: u16) -> Result<Self::Ok, Error> {
        self.serialize_u64(value.into())
    }

    fn serialize_u32(self, value: u32) -> Result<Self::Ok, Error> {
        self.serialize_u64(value.into())
    }

    fn serialize_u64(self, value: u64) -> Result<Self::Ok, Error> {
        let Ok(object) = value.into_pyobject(self.py);
        Ok(object.into_any())
    }

    fn serialize_f32(self, value: f32) -> Result<Self::Ok, Error> {
        self.serialize_f64(value.into())
    }

    fn serialize_f64(self, value: f64) -> Result<Self::Ok, Error> {
        Ok(PyFloat::new(self.py, value).into_any())
    }

    fn serialize_char(self, value: char) -> Result<Self::Ok, Error> {
        self.serialize_str(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<Self::Ok, Error> {
        Ok(PyString::new(self.py, value).into_any())
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Error> {
        let list = PyList::new(self.py, value)?;
        Ok(list.into_any())
    }

    fn serialize_none(self) -> Result<Self::Ok, Error> {
        Ok(self.py.None().into_bound(self.py))
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Error> {
        self.serialize_none()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Error> {
        self.serialize_none()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Error> {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Error> {
        tagged(self.py, variant, value.serialize(self)?)
    }

    fn serialize_seq(self, _length: Option<usize>) -> Result<Seq<'py>, Error> {
        Ok(Seq {
            py: self.py,
            list: PyList::empty(self.py),
            variant: None,
        })
    }

    fn serialize_tuple(self, length: usize) -> Result<Seq<'py>, Error> {
        self.serialize_seq(Some(length))
    }

    fn serialize_tuple_struct(self, _name: &'static str, length: usize) -> Result<Seq<'py>, Error> {
        self.serialize_seq(Some(length))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _length: usize,
    ) -> Result<Seq<'py>, Error> {
        Ok(Seq {
            py: self.py,
            list: PyList::empty(self.py),
            variant: Some(variant),
        })
    }

    fn serialize_map(self, _length: Option<usize>) -> Result<Map<'py>, Error> {
        Ok(Map {
            py: self.py,
            dict: PyDict::new(self.py),
            key: None,
            variant: None,
        })
    }

    fn serialize_struct(self, _name: &'static str, length: usize) -> Result<Map<'py>, Error> {
        self.serialize_map(Some(length))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _length: usize,
    ) -> Result<Map<'py>, Error> {
        Ok(Map {
            py: self.py,
            dict: PyDict::new(self.py),
            key: None,
            variant: Some(variant),
        })
    }

    fn collect_str<T: fmt::Display + ?Sized>(self, value: &T) -> Result<Self::Ok, Error> {
        self.serialize_str(&value.to_string())
    }
}

/// A sequence being built, the content of a tuple variant when `variant`.
struct Seq<'py> {
    py: Python<'py>,
    list: Bound<'py, PyList>,
    variant: Option<&'static str>,
}

impl Seq<'_> {
    fn push<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.list
            .append(value.serialize(Serializer { py: self.py })?)?;
        Ok(())
    }
}

impl<'py> Seq<'py> {
    fn finish(self) -> Result<Bound<'py, PyAny>, Error> {
        match self.variant {
            Some(variant) => tagged(self.py, variant, self.list.into_any()),
            None => Ok(self.list.into_any()),
        }
    }
}

impl<'py> ser::SerializeSeq for Seq<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

impl<'py> ser::SerializeTuple for Seq<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

impl<'py> ser::SerializeTupleStruct for Seq<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

impl<'py> ser::SerializeTupleVariant for Seq<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

/// A map being built, the content of a struct variant when `variant`.
struct Map<'py> {
    py: Python<'py>,
    dict: Bound<'py, PyDict>,
    key: Option<Bound<'py, PyAny>>,
    variant: Option<&'static str>,
}

impl<'py> Map<'py> {
    fn finish(self) -> Result<Bound<'py, PyAny>, Error> {
        match self.variant {
            Some(variant) => tagged(self.py, variant, self.dict.into_any()),
            None => Ok(self.dict.into_any()),
        }
    }

    fn field<T: Serialize + ?Sized>(&mut self, key: &'static str, value: &T) -> Result<(), Error> {
        self.dict
            .set_item(key, value.serialize(Serializer { py: self.py })?)?;
        Ok(())
    }
}

impl<'py> ser::SerializeMap for Map<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        let key = key.serialize(Serializer { py: self.py })?;
        if !key.is_instance_of::<PyString>() {
            return Err(Error("a map key is not a string".to_owned()));
        }
        self.key = Some(key);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .key
            .take()
            .ok_or_else(|| Error("a map value comes before its key".to_owned()))?;
        self.dict
            .set_item(key, value.serialize(Serializer { py: self.py })?)?;
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

impl<'py> ser::SerializeStruct for Map<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.field(key, value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}

impl<'py> ser::SerializeStructVariant for Map<'py> {
    type Ok = Bound<'py, PyAny>;
    type Error = Error;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.field(key, value)
    }

    fn end(self) -> Result<Self::Ok, Error> {
        self.finish()
    }
}
