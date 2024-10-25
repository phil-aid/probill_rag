```mermaid
erDiagram

    %% Entities
    User ||--o{ Patient : "physician_id"
    User ||--o{ Encounter : "physician_id"
    User ||--o{ VersionedEncounterBillingRecord : "biller_id"

    Patient ||--o{ Encounter : "patient_id"

    Encounter ||--o{ VersionedEncounterBillingRecord : "encounter_id"

    VersionedEncounterBillingRecord ||--o{ VersionedEncounterBillingRecord : "parent_id"

    PatientAccess }o--|| User : "biller_id"
    PatientAccess }o--|| Patient : "patient_id"

    Merge }o--|| VersionedEncounterBillingRecord : "source_record_id"
    Merge }o--|| VersionedEncounterBillingRecord : "target_record_id"
    Merge }o--|| User : "merged_by"

    AuditLog }o--|| User : "user_id"
    AuditLog }o--|| Patient : "patient_id"
    AuditLog }o--|| Encounter : "encounter_id"
    AuditLog }o--|| VersionedEncounterBillingRecord : "record_id"

    %% Entity Definitions

    class User {
        int id PK
        string username
        string username_lower
        bytes password
        bytes secret
        string did
        string user_public_key
        string subscription_name
        bool is_active
        datetime creation_date
        string role
    }

    class Patient {
        int id PK
        int physician_id FK
        string name
        datetime date_of_birth
        string medical_record_number
        bool is_active
        datetime creation_date
    }

    class Encounter {
        int id PK
        int patient_id FK
        int physician_id FK
        datetime encounter_date
        string reason
        bool is_active
    }

    class PatientAccess {
        int id PK
        int biller_id FK
        int patient_id FK
        datetime access_granted_date
        bool is_active
    }

    class VersionedEncounterBillingRecord {
        int id PK
        int encounter_id FK
        int biller_id FK
        int version
        string branch
        string content
        int created_by FK
        datetime created_at
        int parent_id FK
        bool is_active
        string message
    }

    class Merge {
        int id PK
        int source_record_id FK
        int target_record_id FK
        int merged_by FK
        datetime merged_at
        string conflict_resolution
    }

    class AuditLog {
        int id PK
        int user_id FK
        int patient_id FK
        int encounter_id FK
        datetime action_time
        string action
        int record_id FK
        string details
    }
```
