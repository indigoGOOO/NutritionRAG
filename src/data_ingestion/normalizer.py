"""Normalizers that map external nutrition datasets into canonical schemas."""

from __future__ import annotations

import re
from typing import Any

from src.data_ingestion.schemas import FoodEntity, ImportBatch, NutrientProfile, SourceMetadata


NUTRIENT_NAME_MAP = {
    "energy": "energy_kcal",
    "energy_kcal": "energy_kcal",
    "calories": "energy_kcal",
    "protein": "protein_g",
    "total lipid (fat)": "fat_g",
    "fat": "fat_g",
    "carbohydrate, by difference": "carbohydrate_g",
    "carbohydrates": "carbohydrate_g",
    "carbohydrate": "carbohydrate_g",
    "fiber, total dietary": "fiber_g",
    "fiber": "fiber_g",
    "sugars, total including nlea": "sugar_g",
    "sugars": "sugar_g",
    "sugar": "sugar_g",
    "sodium, na": "sodium_mg",
    "sodium": "sodium_mg",
    "potassium, k": "potassium_mg",
    "calcium, ca": "calcium_mg",
    "iron, fe": "iron_mg",
}

OFF_NUTRIENT_MAP = {
    "energy-kcal_100g": "energy_kcal",
    "energy_100g": "energy_kj",
    "proteins_100g": "protein_g",
    "fat_100g": "fat_g",
    "carbohydrates_100g": "carbohydrate_g",
    "fiber_100g": "fiber_g",
    "sugars_100g": "sugar_g",
    "sodium_100g": "sodium_g",
    "salt_100g": "salt_g",
}


def canonical_entity_id(name: str, prefix: str = "") -> str:
    value = name.strip().lower()
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if prefix and not value.startswith(f"{prefix}_"):
        return f"{prefix}_{value}"
    return value or "unknown"


def normalize_nutrient_name(name: str) -> str:
    return NUTRIENT_NAME_MAP.get(name.strip().lower(), canonical_entity_id(name))


def normalize_usda_food(record: dict[str, Any], source: SourceMetadata) -> ImportBatch:
    """Normalize one USDA FDC-like JSON record.

    Supports common fields from the API/download JSON: fdcId, description,
    foodCategory, foodNutrients[{nutrientName/name, amount, unitName}].
    """
    fdc_id = str(record.get("fdcId") or record.get("fdc_id") or "")
    name = str(record.get("description") or record.get("name") or "").strip()
    if not name:
        return ImportBatch()

    entity_id = canonical_entity_id(name, "usda")
    metadata = SourceMetadata(
        source=source.source,
        source_type=source.source_type,
        source_url=source.source_url,
        language=source.language or "en",
        confidence=source.confidence,
        updated_at=source.updated_at,
        extra={**source.extra, "fdc_id": fdc_id} if fdc_id else source.extra,
    )
    entity = FoodEntity(
        entity_id=entity_id,
        name=name,
        entity_type="food",
        aliases=[name],
        source=metadata,
    )

    nutrients = {}
    for item in record.get("foodNutrients", []) or record.get("nutrients", []):
        nutrient = item.get("nutrient", {}) if isinstance(item, dict) else {}
        raw_name = (
            item.get("nutrientName")
            or item.get("name")
            or nutrient.get("name")
            or nutrient.get("nutrientName")
            or ""
        )
        amount = item.get("amount") or item.get("value")
        if raw_name and amount not in (None, ""):
            nutrients[normalize_nutrient_name(str(raw_name))] = amount

    profile = NutrientProfile(
        entity_id=entity_id,
        entity_type="food",
        nutrients=nutrients,
        serving_basis="per_100g",
        source=metadata,
    )
    return ImportBatch(entities=[entity], nutrient_profiles=[profile])


def normalize_openfoodfacts_product(record: dict[str, Any], source: SourceMetadata) -> ImportBatch:
    """Normalize one Open Food Facts product JSON object."""
    product_name = str(record.get("product_name") or record.get("product_name_en") or "").strip()
    code = str(record.get("code") or record.get("_id") or "").strip()
    if not product_name:
        return ImportBatch()

    entity_id = f"off_{code}" if code else canonical_entity_id(product_name, "off")
    metadata = SourceMetadata(
        source=source.source,
        source_type=source.source_type,
        source_url=source.source_url or (f"https://world.openfoodfacts.org/product/{code}" if code else ""),
        language=source.language,
        confidence=source.confidence,
        updated_at=source.updated_at,
        extra={**source.extra, "barcode": code} if code else source.extra,
    )
    aliases = [product_name]
    generic_name = str(record.get("generic_name") or "").strip()
    brands = str(record.get("brands") or "").strip()
    if generic_name:
        aliases.append(generic_name)
    if brands:
        aliases.append(f"{brands} {product_name}".strip())

    nutriments = record.get("nutriments", {}) or {}
    nutrients = {}
    for raw_key, target_key in OFF_NUTRIENT_MAP.items():
        value = nutriments.get(raw_key)
        if value not in (None, ""):
            nutrients[target_key] = value

    value = {
        **nutrients,
        "product_name": product_name,
        "brand": brands,
        "barcode": code,
        "ingredients": record.get("ingredients_text") or record.get("ingredients_text_en") or "",
        "categories": record.get("categories") or "",
        "basis": "per_100g",
    }
    profile = NutrientProfile(
        entity_id=entity_id,
        entity_type="branded_food",
        nutrients=value,
        serving_basis="per_100g",
        source=metadata,
    )
    entity = FoodEntity(
        entity_id=entity_id,
        name=product_name,
        entity_type="branded_food",
        aliases=aliases,
        source=metadata,
    )
    return ImportBatch(entities=[entity], nutrient_profiles=[profile])


def merge_batches(batches: list[ImportBatch]) -> ImportBatch:
    merged = ImportBatch()
    for batch in batches:
        merged.entities.extend(batch.entities)
        merged.nutrient_profiles.extend(batch.nutrient_profiles)
        merged.relations.extend(batch.relations)
        merged.documents.extend(batch.documents)
    return merged

