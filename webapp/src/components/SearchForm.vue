<template>
<v-form v-model="valid" @submit.prevent="submit">
    <v-container style="width: 70vw">
        <v-row>
            <v-col>
                <v-text-field
                    v-model="id1"
                    :rules="amountRules"
                    label="From ID"
                    required
                    type="number"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    v-model="id2"
                    :rules="amountRules"
                    label="To ID"
                    required
                    type="number"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row>
            <v-col>
                <v-btn
                color="primary"
                class="mr-4"
                type="submit"
                :disabled="isLoading"
                >
                Get images
                </v-btn>
                <v-progress-circular
                    indeterminate
                    color="primary"
                    v-if="isLoading"
                ></v-progress-circular>
            </v-col>
        </v-row>
    </v-container>
  </v-form>
</template>

<script>
    import axios from 'axios'
    export default {
        data: () => ({
        refCount: 0,
        isLoading: false,
        valid: false,
        id1: '',
        id2: '',
        amountRules: [
            v => !!v || 'Amount is required',
            v => Number.isInteger(v) > 0 || 'Amount must be an integer',
            v => Number(v) > 0 || 'Amount must be a positive integer',
        ],
        }),
        methods : {
            submit(){
                axios({
                    url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/faces?id1='+this.id1+'&id2='+this.id2,
                    method: 'get',
                }).then((response) => {
                    console.log(response)
                    this.$emit('generated', response.data.imgs_bytes, response.data.ids)
                }).catch((e) => {
                    alert('Error')
                    console.log(e)
                })
            },
            setLoading(isLoading) {
                if (isLoading) {
                    this.refCount++;
                    this.isLoading = true;
                } else if (this.refCount > 0) {
                    this.refCount--;
                    this.isLoading = (this.refCount > 0);
                }
            }
        },
        created() {
            axios.interceptors.request.use((config) => {
                this.setLoading(true);
                return config;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });

            axios.interceptors.response.use((response) => {
                this.setLoading(false);
                return response;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });
        }
    }
</script>